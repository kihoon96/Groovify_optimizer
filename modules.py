import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os.path as osp
import math
import torchvision.models.resnet as resnet

# from core.config import cfg
from layers import make_linear_layers, make_conv_layers, make_deconv_layers, LocallyConnected2d, KeypointAttention


from easydict import EasyDict as edict

cfg = edict()

""" Model - MP """
cfg.MD = edict()
cfg.MD.hidden_dim = 147
cfg.MD.seqlen = 256
cfg.MD.mid_frame = 24
cfg.MD.num_layers = 4
cfg.MD.resample = 256
cfg.MD.weight_path = ''
output_dim = 256 -1


class GroovifyNet(nn.Module):
    def __init__(self):
        super().__init__()       
        self.motion_fc_in = nn.Linear(147, cfg.MD.hidden_dim)
        
        self.bpm_projection = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.LayerNorm(32)
        )
        
        self.fusion_attention = RhythmAwareFusion()

        self.motion_mlp = TransMLP(cfg.MD.hidden_dim, cfg.MD.seqlen, cfg.MD.num_layers)
        self.motion_fc_out = nn.Linear(cfg.MD.hidden_dim, cfg.MD.hidden_dim)
        
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.dimension_projector = nn.Linear(cfg.MD.hidden_dim, output_dim)
        self.output_activation = nn.Sigmoid()
        
    def init_weights(self):
        # 추가필요
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def load_weights(self, checkpoint):
        from train_utils import check_data_parallel
        self.load_state_dict(check_data_parallel(checkpoint['model']), strict=False)  

    def forward(self, x, bpm):        
        # pose embedding
        x = self.motion_fc_in(x)
        
        # bpm embedding
        bpm = self.bpm_projection(bpm.unsqueeze(-1))
        
        # fusion attention
        x = self.fusion_attention(x, bpm)
        
        # after fusion mlp
        x = x.permute(0, 2, 1)
        x = self.motion_mlp(x)
        x = x.permute(0, 2, 1)
        x = self.motion_fc_out(x)
        
        x = self.temporal_pool(x.permute(0,2,1)).squeeze(-1)
        x = self.dimension_projector(x)
        x = self.output_activation(x)
        return x

class AdaptiveResampler(nn.Module):
    def __init__(self, output_size=256):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, x):
        B, T, C = x.shape
        x_perm = x.permute(0, 2, 1)
        resampled = F.interpolate(
            x_perm, 
            size=self.output_size,
            mode='linear',
            align_corners=False
        )
        return resampled.permute(0, 2, 1)


class RhythmAwareFusion(nn.Module):
    def __init__(self, pose_dim=147, bpm_dim=32, hidden_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections for pose sequence
        self.q_proj = nn.Linear(pose_dim, hidden_dim)
        self.k_proj = nn.Linear(pose_dim, hidden_dim)
        self.v_proj = nn.Linear(pose_dim, hidden_dim)
        
        # Projection for BPM feature
        self.bpm_proj = nn.Linear(bpm_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, pose_dim)
        
    def forward(self, pose, bpm_emb):
        """
        Args:
            pose: Tensor of shape [B, T, C] where T=256, C=147
            bpm_emb: Tensor of shape [B, 32]
        Returns:
            Tensor of shape [B, T, C]
        """
        B, T, C = pose.shape
        
        # Project pose to query, key, value
        q = self.q_proj(pose).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, T, D]
        k = self.k_proj(pose).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, T, D]
        v = self.v_proj(pose).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, T, D]
        
        # Project and broadcast BPM embedding to influence attention
        bpm_feature = self.bpm_proj(bpm_emb)  # [B, hidden_dim]
        bpm_feature = bpm_feature.reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, 1, D]
        
        # Modify keys with BPM information
        k = k + bpm_feature  # Broadcasting: [B, H, T, D] + [B, H, 1, D] -> [B, H, T, D]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, T, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, T, T]
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, v)  # [B, H, T, D]
        
        # Reshape and project back to original dimensions
        out = out.permute(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_dim)  # [B, T, hidden_dim]
        out = self.out_proj(out)  # [B, T, C]
        return out

# # Attention Fusion
# class RhythmAwareFusion(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.query = nn.Linear(147, 64) # 24x6 rot + 3 trans
#         self.key = nn.Linear(32, 64)
        
#     def forward(self, pose, bpm_emb):
#         import pdb; pdb.set_trace()
#         pose = pose.permute(0,2,1)
#         query = self.query(pose) # [B, 147, 64]
#         key = self.key(bpm_emb).unsqueeze(1)
        
#         attn_scores = torch.matmul(query, key.transpose(-1, -2)) # [B, 147, 1]
#         attn_weights = torch.softmax(attn_scores, dim=1) # [B, 147, 1]
#         fused = torch.matmul(attn_weights.transpose(1,2), pose)  # [B, 1, 256]
#         import pdb; pdb.set_trace()
#         return fused.squeeze(1)  # [B, 256]

class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class MLPblock(nn.Module):
    def __init__(self, dim, seq, use_norm=True):
        super().__init__()
        self.fc0 = Temporal_FC(seq)

        layernorm_axis = 'spatial'
        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x):
        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_
        return x


class TransMLP(nn.Module):
    def __init__(self, dim, seq, num_layers):
        super().__init__()
        self.mlps = nn.Sequential(*[
            MLPblock(dim, seq)
            for i in range(num_layers)])

    def forward(self, x):
        x = self.mlps(x)
        return x



class BinarySplitDecoder(nn.Module):
    def __init__(self, tree_depth = 8):
        super().__init__()
        self.depth = tree_depth
        # self.split_params = nn.Sequential(
        #     nn.Linear(cfg.MD.hidden_dim, 2**tree_depth - 1),
        #     nn.Sigmoid()
        # )
        
    def build_tree(self, alphas):
        # Initialize timing matrix [Batch, Nodes]
        nodes = torch.zeros(alphas.shape[0], 2**self.depth).to('cuda')
        
        # Root node covers full sequence
        nodes[:,0] = 1.0
        alpha_idx = 0  
        for d in range(self.depth - 1):
            level_nodes = 2**d
            level_start = 2**d - 1
            for n in range(level_nodes):
                node_idx = level_start + n

                left = 2**(d+1) - 1 + 2*n
                right = left + 1
                # Use sequential indexing for alphas
                alpha = alphas[:, alpha_idx]    
                alpha_idx += 1
                nodes[:, left] = nodes[:, node_idx] * alpha
                nodes[:, right] = nodes[:, node_idx] * (1 - alpha)
        return nodes[:, -2**self.depth :] # Return leaf nodes

    def forward(self, x):
        # alpha_params = self.split_params(x)
        return self.build_tree(x)

def hierarchical_interpolate(sequence, leaf_weights):
    batch, time, feat = sequence.shape
    
    # Compute cumulative distribution
    cum_weights = torch.cumsum(leaf_weights, dim=1)
    
    # Generate sampling grid
    grid = torch.linspace(0, 1, time).to(sequence.device)
    grid = grid.view(1, -1, 1).expand(batch, -1, 1)
    
    import pdb; pdb.set_trace()
    # Find segment indices
    indices = torch.searchsorted(cum_weights.unsqueeze(-1), grid)
    lower = torch.clamp(indices-1, 0)
    upper = torch.clamp(indices, max=leaf_weights.shape[1]-1)
    
    # Linear interpolation weights
    lower_weight = (cum_weights[...,upper] - grid) 
    upper_weight = (grid - cum_weights[...,lower])
    total_weight = (cum_weights[...,upper] - cum_weights[...,lower]) + 1e-8
    
    # Gather features
    lower_feat = torch.gather(sequence, 1, lower.expand(-1,-1,feat))
    upper_feat = torch.gather(sequence, 1, upper.expand(-1,-1,feat))
    
    return (lower_weight/total_weight)*lower_feat + (upper_weight/total_weight)*upper_feat



