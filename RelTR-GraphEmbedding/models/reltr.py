# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import pickle
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib as mpl
import sys
import os
import argparse
import pandas as pd
from scipy import stats
import matplotlib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pykeen.models import ERModel
from pykeen.nn.representation import Embedding
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
import numpy as np
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision import models
import warnings
import gc
import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer


def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

class CustomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scaling = self.head_dim ** -0.5

    def forward(self, query, key, value, mask=None):
        # Linear projections
        Q = self.query_proj(query)  # (batch_size, seq_len_q, embed_dim)
        K = self.key_proj(key)      # (batch_size, seq_len_k, embed_dim)
        V = self.value_proj(value)  # (batch_size, seq_len_v, embed_dim)

        # Reshape for multi-head attention
        batch_size, embed_dim= Q.size()
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_q, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)         # (batch, heads, seq_k, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)         # (batch, heads, seq_v, head_dim)

        # Attention scores
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling  # (batch, heads, seq_q, seq_k)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)  # Normalize scores
        attn_output = torch.matmul(attn_weights, V)         # Weighted sum (batch, heads, seq_q, head_dim)

        # Concatenate and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, embed_dim)
        output = self.out_proj(attn_output)  # (batch, seq_len_q, embed_dim)

        return output, attn_weights


class RelTR(nn.Module):
    """ RelTR: Relation Transformer for Scene Graph Generation """
    def __init__(self, backbone, transformer, num_classes, num_rel_classes, num_entities, num_triplets, aux_loss=False, matcher=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of entity classes
            num_entities: number of entity queries
            num_triplets: number of coupled subject/object queries
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.embeddings = nn.Parameter(self.setup_rgcn())
        self.num_rel_classes = num_rel_classes
        self.num_entities = num_entities
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.entity_embed = nn.Embedding(num_entities, hidden_dim*2)
        self.triplet_embed = nn.Embedding(num_triplets, hidden_dim*3)
        self.so_embed = nn.Embedding(2, hidden_dim) # subject and object encoding

        # entity prediction
        self.entity_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.entity_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # mask head
        self.so_mask_conv = nn.Sequential(torch.nn.Upsample(size=(28, 28)),
                                          nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=3, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(64),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(32))
        self.so_mask_fc = nn.Sequential(nn.Linear(2048, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, 128))

        # predicate classification
        self.rel_class_embed = MLP(hidden_dim*2+128, hidden_dim, num_rel_classes + 1, 2)

        # subject/object label classfication and box regression
        self.sub_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Commonsense branch
        self.attention = CustomAttention(512, 1)
        self.cs_rel_pred = MLP(512, 128, num_rel_classes + 1, 2)

        # Fusion MLP takes a concatenation of [visual_relation_logits, cs_relation_logits]
        fusion_in_dim = 2 * (num_rel_classes + 1)
        self.fusion_mlp = MLP(fusion_in_dim, 256, num_rel_classes + 1, 2)


    def setup_rgcn(self):
        CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
            'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
            'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
            'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
            'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
            'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
            'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
            'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
            'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
            'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
            'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
            'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
            'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
            'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
        
        CLASSES_ = ["N/A","airplane.n.01", "animal.n.01", "arm.n.01", "bag.n.01", "banana.n.01", "basket.n.01", "beach.n.01", "bear.n.01", "bed.n.01", "bench.n.01", "motorcycle.n.01", "bird.n.01", "board.n.01", "boat.n.01", "book.n.01", "boot.n.01", "bottle.n.01", "bowl.n.01", "box.n.01", "male_child.n.01", "branch.n.01", "building.n.01", "bus.n.01", "cabinet.n.01", "cap.n.01", "car.n.01", "cat.n.01", "chair.n.01", "child.n.01", "clock.n.01", "coat.n.01", "counter.n.01", "cow.n.01", "cup.n.01", "curtain.n.01", "desk.n.01", "dog.n.01", "door.n.01", "drawer.n.01", "ear.n.01", "elephant.n.01", "engine.n.01", "eye.n.01", "face.n.01", "fence.n.01", "finger.n.01", "flag.n.01", "flower.n.01", "food.n.01", "fork.n.01", "fruit.n.01", "giraffe.n.01", "girl.n.01", "glass.n.01", "baseball_glove.n.01", "guy.n.01", "hair.n.01", "hand.n.01", "handle.n.01", "hat.n.01", "head.n.01", "helmet.n.01", "hill.n.01", "horse.n.01", "house.n.01", "jacket.n.01", "jean.n.01", "child.n.01", "kite.n.01", "lady.n.01", "lamp.n.01", "laptop.n.01", "leaf.n.01", "leg.n.01", "letter.n.01", "light.n.01", "logo.n.01", "man.n.01", "man.n.01", "motorcycle.n.01", "mountain.n.01", "mouth.n.01", "neck.n.01", "nose.n.01", "number.n.01", "orange.n.01", "pant.n.01", "paper.n.01", "paw.n.01", "people.n.01", "person.n.01", "telephone.n.01", "pillow.n.01", "pizza.n.01", "airplane.n.01", "plant.n.01", "home_plate.n.01", "player.n.01", "pole.n.01", "post.n.01", "pot.n.01", "racket.n.01", "railing.n.01", "rock.n.01", "roof.n.01", "room.n.01", "screen.n.01", "seat.n.01", "sheep.n.01", "shelf.n.01", "shirt.n.01", "shoe.n.01", "short.n.01", "sidewalk.n.01", "sign.n.01", "sink.n.01", "skateboard.n.01", "ski.n.01", "skier.n.01", "shoe.n.01", "snow.n.01", "sock.n.01", "base.n.08", "street.n.01", "board.n.01", "table.n.01", "tail.n.01", "necktie.n.01", "tile.n.01", "tire.n.01", "toilet.n.01", "towel.n.01", "tower.n.01", "path.n.04", "train.n.01", "tree.n.01", "truck.n.01", "trunk.n.01", "umbrella.n.01", "vase.n.01", "vegetable.n.01", "vehicle.n.01", "wave.n.01", "wheel.n.01", "window.n.01", "windshield.n.01", "wing.n.01", "wire.n.01", "woman.n.01", "zebra.n.01"]
        if(len(CLASSES)!=len(CLASSES_)):
            print('Lengths not equal')
            raise ValueError
        objectTracker = set()
        with open ('wn-total-llm.txt', 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                s,p,o = line.strip().split('\t')
                objectTracker.add(s)
                objectTracker.add(o)
        
        object_to_idx = {}
        idx_to_object = {}
        for i, obj in enumerate(sorted(list(objectTracker))):
            object_to_idx[obj] = i
            idx_to_object[i] = obj
        # print('Object to idx',object_to_idx)
        model = torch.load('./results-total-256-run/trained_model.pkl', weights_only=False)

        entity_representation_modules = model.entity_representations[0]()
        embeddings = torch.zeros(256)
        embeddings = embeddings.reshape(1,256)
        embeddings.to('cpu')
        entity_representation_modules = entity_representation_modules.to('cpu')
        for i in range(1,151):
            if(CLASSES_[i] in object_to_idx.keys()):
                objIdx = object_to_idx[CLASSES_[i]]
                try:
                    embed_val = entity_representation_modules[objIdx]
                except:
                    raise ValueError
                embed_val = embed_val.reshape(1,256)
                embeddings=torch.cat((embeddings,embed_val),0)
            else:
                embeddings = torch.cat((embeddings,torch.rand(1,256)),0)
                raise ValueError
                

        embeddings = torch.cat((embeddings,torch.zeros(1,256)),0)
        print(embeddings.shape,"Embeddings shape now loaded")
        return embeddings
    
    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the entity classification logits (including no-object) for all entity queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": the normalized entity boxes coordinates for all entity queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "sub_logits": the subject classification logits
               - "obj_logits": the object classification logits
               - "sub_boxes": the normalized subject boxes coordinates
               - "obj_boxes": the normalized object boxes coordinates
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs, hs_t, so_masks, _ = self.transformer(self.input_proj(src), mask, self.entity_embed.weight,
                                                 self.triplet_embed.weight, pos[-1], self.so_embed.weight)
        so_masks = so_masks.detach()
        so_masks = self.so_mask_conv(so_masks.view(-1, 2, src.shape[-2],src.shape[-1])).view(hs_t.shape[0], hs_t.shape[1], hs_t.shape[2],-1)
        so_masks = self.so_mask_fc(so_masks)

        hs_sub, hs_obj = torch.split(hs_t, self.hidden_dim, dim=-1)

        outputs_class = self.entity_class_embed(hs)
        outputs_coord = self.entity_bbox_embed(hs).sigmoid()

        outputs_class_sub = self.sub_class_embed(hs_sub)
        outputs_coord_sub = self.sub_bbox_embed(hs_sub).sigmoid()

        outputs_class_obj = self.obj_class_embed(hs_obj)
        outputs_coord_obj = self.obj_bbox_embed(hs_obj).sigmoid()

        outputs_class_rel = self.rel_class_embed(torch.cat((hs_sub, hs_obj, so_masks), dim=-1))

        rel1_logits = outputs_class_rel  # Final relation logits from visual branch; shape: [batch, num_queries, num_rel_classes+1]
    
        # Commonsense branch: create alternative relation logits
        # Get predicted subject/object indices 
        indices_obj = outputs_class_obj.argmax(dim=-1)  # [6,batch, num_queries]
        indices_sub = outputs_class_sub.argmax(dim=-1)  # [6,batch, num_queries]

        # Ensure the indices and your stored embeddings are on the same device.
        indices_obj = indices_obj.to(self.embeddings.device)
        indices_sub = indices_sub.to(self.embeddings.device)
        
        # Lookup your common-sense embeddings.
        embed_obj = self.embeddings[indices_obj]   # shape: [6,batch, num_queries, embed_dim]
        embed_sub = self.embeddings[indices_sub]     # shape: [6,batch, num_queries, embed_dim]
        
        # Concatenate the two embeddings along the feature dimension.
        embed_cat = torch.cat((embed_sub, embed_obj), dim=-1)  # shape: [6,batch, num_queries, 2*embed_dim]
        embed_cat_flat = embed_cat.view(-1, 512)
        visual_feats = torch.cat((hs_sub, hs_obj), dim=-1)  # shape: [6, batch, num_queries, 2*embed_dim]
        visual_feats_flat = visual_feats.view(-1, 512)
        # print('visual_feats', visual_feats.shape) # shape: [batch, num_queries, 512]
        attn_output, _ = self.attention(visual_feats_flat, embed_cat_flat, embed_cat_flat, None)
        attn_output = attn_output.view(visual_feats.shape[0],visual_feats.shape[1], visual_feats.shape[2], 512)
        cs_rel_logits = self.cs_rel_pred(attn_output)  # shape: [6, batch, num_queries, num_rel_classes+1]
        
        # Fusion via the MLP: concatenate the two sets of relation logits
        fused_input = torch.cat((rel1_logits, cs_rel_logits), dim=-1)  # shape: [6,batch, num_queries, 2*(num_rel_classes+1)]
        fused_rel_logits = self.fusion_mlp(fused_input)  # shape: [6,batch, num_queries, num_rel_classes+1]
    
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
           'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
           'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
           'rel_logits': fused_rel_logits[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                outputs_class_obj, outputs_coord_obj, fused_rel_logits
            )
        return out
        
        

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                      outputs_class_obj, outputs_coord_obj, outputs_class_rel):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'sub_logits': c, 'sub_boxes': d, 'obj_logits': e, 'obj_boxes': f,
                 'rel_logits': g}
                for a, b, c, d, e, f, g in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_sub[:-1],
                                               outputs_coord_sub[:-1], outputs_class_obj[:-1], outputs_coord_obj[:-1],
                                               outputs_class_rel[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for RelTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_rel_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.num_rel_classes = 51 if num_classes == 151 else 31 # Using entity class numbers to adapt rel class numbers
        empty_weight_rel = torch.ones(num_rel_classes+1)
        empty_weight_rel[-1] = self.eos_coef
        self.register_buffer('empty_weight_rel', empty_weight_rel)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Entity/subject/object Classification loss
        """
        assert 'pred_logits' in outputs

        pred_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices[0])
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices[0])])
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o

        sub_logits = outputs['sub_logits']
        obj_logits = outputs['obj_logits']

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 0]] for t, (_, J) in zip(targets, indices[1])])
        target_relo_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 1]] for t, (_, J) in zip(targets, indices[1])])

        target_sub_classes = torch.full(sub_logits.shape[:2], self.num_classes, dtype=torch.int64, device=sub_logits.device)
        target_obj_classes = torch.full(obj_logits.shape[:2], self.num_classes, dtype=torch.int64, device=obj_logits.device)

        target_sub_classes[rel_idx] = target_rels_classes_o
        target_obj_classes[rel_idx] = target_relo_classes_o

        target_classes = torch.cat((target_classes, target_sub_classes, target_obj_classes), dim=1)
        src_logits = torch.cat((pred_logits, sub_logits, obj_logits), dim=1)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction='none')

        loss_weight = torch.cat((torch.ones(pred_logits.shape[:2]).to(pred_logits.device), indices[2]*0.5, indices[3]*0.5), dim=-1)
        losses = {'loss_ce': (loss_ce * loss_weight).sum()/self.empty_weight[target_classes].sum()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(pred_logits[idx], target_classes_o)[0]
            losses['sub_error'] = 100 - accuracy(sub_logits[rel_idx], target_rels_classes_o)[0]
            losses['obj_error'] = 100 - accuracy(obj_logits[rel_idx], target_relo_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['rel_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["rel_annotations"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the entity/subject/object bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices[0])
        pred_boxes = outputs['pred_boxes'][idx]
        target_entry_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices[0])], dim=0)

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 0]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        target_relo_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 1]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        rels_boxes = outputs['sub_boxes'][rel_idx]
        relo_boxes = outputs['obj_boxes'][rel_idx]

        src_boxes = torch.cat((pred_boxes, rels_boxes, relo_boxes), dim=0)
        target_boxes = torch.cat((target_entry_boxes, target_rels_boxes, target_relo_boxes), dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_relations(self, outputs, targets, indices, num_boxes, log=True):
        """Compute the predicate classification loss
        """
        assert 'rel_logits' in outputs

        src_logits = outputs['rel_logits']
        idx = self._get_src_permutation_idx(indices[1])
        target_classes_o = torch.cat([t["rel_annotations"][J,2] for t, (_, J) in zip(targets, indices[1])])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)

        losses = {'loss_rel': loss_ce}
        if log:
            losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'relations': self.loss_relations
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        self.indices = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"])+len(t["rel_annotations"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels' or loss == 'relations':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):

    num_classes = 151 if args.dataset != 'oi' else 289 # some entity categories in OIV6 are deactivated.
    num_rel_classes = 51 if args.dataset != 'oi' else 31

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    matcher = build_matcher(args)
    model = RelTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_rel_classes = num_rel_classes,
        num_entities=args.num_entities,
        num_triplets=args.num_triplets,
        aux_loss=args.aux_loss,
        matcher=matcher)

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_rel'] = args.rel_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', "relations"]

    criterion = SetCriterion(num_classes, num_rel_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

