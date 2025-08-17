import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention that reduces peak memory usage through:
    1. Chunked computation
    2. Linear attention option
    3. Kernel-based computation
    """
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Single projection for efficient memory usage
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def _chunk_compute_attention(self, q, k, v, chunk_size=1024):
        """Compute attention in chunks to reduce memory footprint"""
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Reshape for efficient computation
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Initialize output tensor
        output = torch.zeros_like(v)
        
        # Process in chunks to avoid materializing full attention matrix
        for i in range(0, seq_len, chunk_size):
            end_idx = min(seq_len, i + chunk_size)
            
            # Current query chunk
            q_chunk = q[:, :, i:end_idx]
            
            # Compute scaled dot-product for this chunk only
            attn_weights = torch.matmul(q_chunk, k.transpose(-1, -2)) * self.scale
            
            # Apply softmax and dropout
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention weights to values
            chunk_output = torch.matmul(attn_weights, v)
            output[:, :, i:end_idx] = chunk_output
            
            # Free memory
            del attn_weights, chunk_output
            torch.cuda.empty_cache()
            
        # Reshape back to original format
        output = output.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
        return output
    
    def _kernel_attention(self, q, k, v):
        """
        Kernel-based attention implementation with linear complexity O(N)
        instead of quadratic O(N²)
        """
        # Apply feature map approximation (ReLU feature map)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Compute KV matrix first (batch, heads, head_dim, head_dim)
        kv = torch.matmul(k.transpose(1, 2), v)  # (B, H, D, D)
        
        # Then multiply with query (avoiding NxN attention matrix)
        context = torch.matmul(q, kv)  # (B, N, H, D)
        
        # Normalize
        normalizer = torch.matmul(q, k.sum(dim=1).unsqueeze(-1))
        context = context / (normalizer + 1e-5)
        
        return context
    
    def forward(self, x, use_kernel_attn=False, chunk_size=512):
        batch_size, seq_len, dim = x.shape
        
        # Compute QKV projections in a memory-efficient way
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Choose attention implementation based on sequence length
        if use_kernel_attn:
            # Use kernel attention (more memory efficient for longer sequences)
            attn_output = self._kernel_attention(q, k, v)
        else:
            # Use chunked attention (balanced approach)
            attn_output = self._chunk_compute_attention(q, k, v, chunk_size)
        
        # Reshape and apply output projection
        attn_output = attn_output.reshape(batch_size, seq_len, dim)
        output = self.proj(attn_output)
        
        return output


class MemoryEfficientEncoderLayer(nn.Module):
    """Memory-efficient transformer encoder layer"""
    def __init__(self, dim, num_heads=8, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.attn = MemoryEfficientAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Smaller MLP dimension ratio for memory efficiency
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(),  # ReLU is more memory-efficient than GELU
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, use_kernel_attn=False, chunk_size=512):
        # Use checkpointing to save memory during training
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0])
            return custom_forward
        
        # Apply attention and MLP in memory-efficient manner
        x = x + self.dropout(checkpoint(create_custom_forward(self.attn), 
                                         self.norm1(x), use_kernel_attn, chunk_size))
        x = x + self.dropout(checkpoint(create_custom_forward(self.mlp), self.norm2(x)))
        return x


class CompactEmbeddingEncoder(nn.Module):
    """
    Memory-efficient encoder for compressing sequence of embeddings into a single embedding
    """
    def __init__(
        self, 
        input_dim=512,
        hidden_dim=384,  # Reduced dimension
        num_layers=2,    # Fewer layers
        num_heads=4,     # Fewer heads
        dropout=0.1,
        use_checkpointing=True
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.02)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            MemoryEfficientEncoderLayer(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=2,  # Reduced ratio
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Projection for final embedding
        self.pooling = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Optional projection back to original dimension
        self.output_projection = nn.Linear(hidden_dim, input_dim) if hidden_dim != input_dim else None
        
        self.use_checkpointing = use_checkpointing
        
    def forward(self, x, use_kernel_attn=True):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Apply encoder layers
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                x = checkpoint(layer, x, use_kernel_attn)
            else:
                x = layer(x, use_kernel_attn)
        
        # Global pooling to compress sequence
        attn_weights = F.softmax(
            torch.matmul(x, torch.mean(x, dim=1, keepdim=True).transpose(-2, -1)), 
            dim=1
        )
        pooled = torch.matmul(attn_weights.transpose(-2, -1), x)
        
        # Apply final projection
        compressed = self.pooling(pooled)
        
        # Project back to original dimension if needed
        if self.output_projection is not None:
            compressed = self.output_projection(compressed)
            
        return compressed
