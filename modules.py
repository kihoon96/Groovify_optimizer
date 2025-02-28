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
cfg.MD.hidden_dim = 144
cfg.MD.seqlen = 49
cfg.MD.mid_frame = 24
cfg.MD.num_layers = 4
cfg.MD.weight_path = ''


class MDNet(nn.Module):
    def __init__(self):
        super(MDNet, self).__init__()
        self.motion_fc_in = nn.Linear(cfg.MD.hidden_dim, cfg.MD.hidden_dim)
        self.motion_mlp = TransMLP(cfg.MD.hidden_dim, cfg.MD.seqlen, cfg.MD.num_layers)
        self.motion_fc_out = nn.Linear(cfg.MD.hidden_dim, cfg.MD.hidden_dim)
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def load_weights(self, checkpoint):
        from train_utils import check_data_parallel
        self.load_state_dict(check_data_parallel(checkpoint['model']), strict=False)  

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.motion_fc_in(x)
        x = x.permute(0, 2, 1)
        x = self.motion_mlp(x)
        x = x.permute(0, 2, 1)
        x = self.motion_fc_out(x)
        return x

class OptimizedMDNet(nn.Module):
    def __init__(self, output_dim):
        super(OptimizedMDNet, self).__init__()
        self.base_network = MDNet()
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.dimension_projector = nn.Linear(cfg.MD.hidden_dim, output_dim)
        self.output_activation = nn.Sigmoid()
        
    def forward(self, x):
        features = self.base_network(x)
        pooled = self.temporal_pool(features.permute(0,2,1)).squeeze(-1)
        projected = self.dimension_projector(pooled)
        return self.output_activation(projected)


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
    def __init__(self, tree_depth):
        super().__init__()
        self.depth = tree_depth
        self.split_params = nn.Sequential(
            nn.Linear(cfg.MD.hidden_dim, 2**tree_depth - 1),
            nn.Sigmoid()
        )
        
    def build_tree(self, alphas):
        # Initialize timing matrix [Batch, Nodes]
        nodes = torch.zeros(alphas.shape[0], 2**self.depth)
        
        # Root node covers full sequence
        nodes[:,0] = 1.0
        
        for d in range(self.depth):
            level_nodes = 2**d
            for n in range(level_nodes):
                parent_idx = 2**d - 1 + n
                left = 2**(d+1) - 1 + 2*n
                right = left + 1
                alpha = alphas[:,parent_idx]
                nodes[:,left] = nodes[:,parent_idx] * alpha
                nodes[:,right] = nodes[:,parent_idx] * (1 - alpha)
        return nodes[:, -2**self.depth :] # Return leaf nodes

    def forward(self, x):
        alpha_params = self.split_params(x)
        return self.build_tree(alpha_params)

def hierarchical_interpolate(sequence, leaf_weights):
    batch, time, feat = sequence.shape
    
    # Compute cumulative distribution
    cum_weights = torch.cumsum(leaf_weights, dim=1)
    
    # Generate sampling grid
    grid = torch.linspace(0, 1, time).to(sequence.device)
    grid = grid.view(1, -1, 1).expand(batch, -1, 1)
    
    # Find segment indices
    indices = torch.searchsorted(cum_weights, grid)
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



