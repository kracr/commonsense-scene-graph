import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention module using carrier tokens to facilitate
    global information exchange between local windows.
    """
    def __init__(self, dim, num_heads=8, window_size=10, carrier_tokens=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.carrier_tokens = carrier_tokens
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections for queries, keys, values
        self.qkv = nn.Linear(dim, dim * 3)
        
        # Projection for carrier token initialization
        self.carrier_init = nn.Linear(dim, dim)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # Split into local windows
        num_windows = seq_len // self.window_size
        # Pad if necessary
        if seq_len % self.window_size != 0:
            pad_len = self.window_size - (seq_len % self.window_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            num_windows = x.shape[1] // self.window_size
            
        # Reshape to [batch, num_windows, window_size, dim]
        windows = x.view(batch_size, num_windows, self.window_size, dim)
        
        # Initialize carrier tokens for each window
        carrier_tokens = self.carrier_init(windows.mean(dim=2)).unsqueeze(2)
        carrier_tokens = carrier_tokens.expand(-1, -1, self.carrier_tokens, -1)
        
        # Concatenate carrier tokens with window tokens
        combined = torch.cat([
            windows.view(batch_size, num_windows * self.window_size, dim),
            carrier_tokens.reshape(batch_size, num_windows * self.carrier_tokens, dim)
        ], dim=1)
        
        # Apply self-attention
        qkv = self.qkv(self.norm(combined)).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Create attention mask for local windows + their carrier tokens
        mask = torch.zeros(combined.shape[1], combined.shape[1], device=x.device)
        for i in range(num_windows):
            # Local window tokens can attend to their window and carrier tokens
            start_idx = i * self.window_size
            end_idx = (i+1) * self.window_size
            # Window tokens
            mask[start_idx:end_idx, start_idx:end_idx] = 1
            # Carrier tokens for this window
            carrier_start = num_windows * self.window_size + i * self.carrier_tokens
            carrier_end = num_windows * self.window_size + (i+1) * self.carrier_tokens
            mask[start_idx:end_idx, carrier_start:carrier_end] = 1
            mask[carrier_start:carrier_end, start_idx:end_idx] = 1
            
            # Carrier tokens can attend to all carrier tokens (global)
            mask[carrier_start:carrier_end, num_windows*self.window_size:] = 1
        
        # Apply mask
        attn = attn.masked_fill(mask.bool().unsqueeze(0).unsqueeze(0) == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention weights
        out = (attn @ v).transpose(1, 2).reshape(batch_size, -1, dim)
        out = self.proj(out)
        
        # Extract window tokens and reshape back
        window_tokens = out[:, :num_windows*self.window_size, :].view(
            batch_size, num_windows, self.window_size, dim
        ).reshape(batch_size, num_windows*self.window_size, dim)
        
        # Trim padding if necessary
        if seq_len % self.window_size != 0:
            window_tokens = window_tokens[:, :seq_len, :]
        
        return window_tokens


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1, window_size=10, carrier_tokens=4):
        super().__init__()
        self.attn = HierarchicalAttention(dim, num_heads, window_size, carrier_tokens)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)


class HierarchicalTransformerEncoder(nn.Module):
    def __init__(
        self, 
        input_dim=512,
        hidden_dim=512, 
        num_encoder_layers=4,
        num_heads=8,
        dropout=0.1,
        window_size=10,
        carrier_tokens=4
    ):
        super().__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                window_size=window_size,
                carrier_tokens=carrier_tokens
            ) for _ in range(num_encoder_layers)
        ])
        
        # Bottleneck - compresses sequence dimension to 1
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        
    def compress(self, x):
        # Compress sequence to single embedding
        pooled = torch.mean(x, dim=1, keepdim=True)
        return self.bottleneck(pooled)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Embed input
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Apply bottleneck to compress to single token
        compressed = self.compress(x)
        
        return compressed
