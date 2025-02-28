import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os.path as osp
import math
import torchvision.models.resnet as resnet

from core.config import cfg
from models.layer import make_linear_layers, make_conv_layers, make_deconv_layers, LocallyConnected2d, KeypointAttention


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


import torch
import torch.nn as nn
from torch.fft import rfft, rfftfreq
from pytorch_metric_learning.losses import NTXentLoss

def y_energy(fftsignal, bq):
    bqs = [bq/4, 2* bq/4, 3*bq/4, bq, bq*3/2, bq*2, bq*3, bq*4]
    bqs = [bq for bq in bqs if bq <= 5]
    bqs = torch.tensor(bqs, device='cuda')
        
    signal_length = fftsignal.shape[0] * 2 # rfft so x2

    frequencies = torch.fft.rfftfreq(signal_length, d=1/30)

    indices = torch.where((frequencies > 0) & (frequencies <= 5))
    
    y_energy = 0.0

    for i in indices[0]:
        distance = torch.min(torch.abs(frequencies[i] - bqs))
        total_power = torch.sum(torch.abs(fftsignal))
        y_energy += fftsignal[i] * torch.exp(distance*10) / total_power
    return y_energy

def y_energy_batch(fftsignal_batch, bq_batch):
    # Create bqs for each batch item
    bqs = torch.stack([torch.tensor([bq/4, 2*bq/4, 3*bq/4, bq, bq*3/2, bq*2, bq*3, bq*4]) 
                           for bq in bq_batch]).to('cuda')

    #bqs 원래 0~5 마스킹해야하지만 안함
    
    # Get signal length and frequencies
    signal_length = fftsignal_batch.shape[1] * 2 - 2
    frequencies = torch.fft.rfftfreq(signal_length, d=1/30).to('cuda')
    
    # Create frequency mask
    freq_mask = (frequencies > 0) & (frequencies <= 5)
    masked_frequencies = frequencies[freq_mask]
    
    # Compute distances for all frequencies and bqs
    freq_expanded = masked_frequencies.unsqueeze(0).unsqueeze(2)  # [1, freq, 1]
    bqs_expanded = bqs.unsqueeze(1)  # [batch, 1, bqs]
    distances = torch.min(torch.abs(freq_expanded - bqs_expanded), dim=2).values  # [batch, freq]
    
    # Get masked signal and compute total power
    masked_signal = fftsignal_batch[:, freq_mask]
    total_power = torch.sum(torch.abs(fftsignal_batch), dim=1, keepdim=True)
    
    # Compute final energy
    y_energy = torch.sum(masked_signal * torch.exp(distances*10) / total_power, dim=1)
    
    return y_energy

#use example
for pkl, wav in zip(tqdm(sorted_joint), sorted_wav):
    pkl = np.load(os.path.join(pkl_path, pkl), allow_pickle=True)
    try:
        joint_rot = pkl['pose'].reshape(-1,24,3)
        trans = pkl['trans']
    except:
        joint_rot = pkl['q'].reshape(-1,24,3)
        trans = pkl['pos']

    joint3d = smpl.forward(torch.Tensor(joint_rot.reshape(-1,24,3))[None,:], torch.Tensor(trans)[None,:])[0]


b, s, c = model_out.shape
model_x = model_out[:, :, :3]
model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))
model_xp = self.smpl.forward(model_q, model_x)

model_out_v_xp = model_xp[:, 1:] - model_xp[:, :-1]

model_out_v_fft = model_out_v_xp

N = 9* model_xp.shape[1]
#case1 vel - adoption nowon
model_out_padded = F.pad(model_out_a_fft, (0,0,0,N), mode='constant')
#case2 accel
model_out_padded = F.pad(model_out_v_fft, (0,0,0,N), mode='constant')
#case3 forward
model_out_padded = F.pad(model_xp, (0,0,0,N), mode='constant')

rfft_model_out = torch.fft.rfft(model_out_padded, dim=1)
out_magnitude = torch.sqrt(rfft_model_out.real**2 + rfft_model_out.imag**2 + 1e-10)
fft_loss = y_energy_batch(out_magnitude.sum(dim=2), gt_bpms/60.0)


class RhythmLoss(nn.Module):
    def __init__(self, base_freq=1.0, num_bands=8, temp=0.1):
        super().__init__()
        self.freq_weights = nn.Parameter(torch.ones(num_bands))
        self.base_freq = base_freq
        self.temp = temp
        self.contrastive = NTXentLoss(temperature=temp)  # Contrastive regularization
        
    def forward(self, fftsignal_batch, bq_batch, motion_features):
        # Spectral attention with learnable weights
        rhythm_energy = self.spectral_attention(fftsignal_batch, bq_batch)
        
        # Contrastive regularization in motion feature space
        contrast_loss = self.contrastive(motion_features)
        
        # Temporal consistency from DTW alignment [5][7]
        dtw_loss = self.dtw_alignment_loss(fftsignal_batch)
        
        return rhythm_energy + 0.3*contrast_loss + 0.2*dtw_loss

    def spectral_attention(self, fftsignal, bq_tensor):
        # Multi-band frequency masking with learnable importance
        bq_factors = torch.stack([bq_tensor * (2**i) for i in range(4)], dim=1)
        freq_bins = self._create_adaptive_bins(bq_factors)
        
        # Focal frequency weighting [11]
        band_energies = []
        for i, (low, high) in enumerate(freq_bins):
            mask = (self.frequencies >= low) & (self.frequencies <= high)
            band_energy = torch.mean(torch.abs(fftsignal[:,mask]) * self.freq_weights[i])
            band_energies.append(band_energy)
            
        return sum(band_energies) / len(band_energies)

    def dtw_alignment_loss(self, fft_features):
        # Differentiable DTW alignment [5][7]
        alignment_cost = 0
        for i in range(fft_features.shape[0]-1):
            alignment_cost += torch.mean(
                SoftDTW(fft_features[i], fft_features[i+1], gamma=1.0)
            )
        return alignment_cost

class FocalFrequencyLoss(nn.Module):  # From [11]
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        freq_weight = self._get_freq_weight(pred_fft, target_fft)
        loss = torch.mean(freq_weight * torch.abs(pred_fft - target_fft)**2)
        return self.loss_weight * loss

class MotionRegularizer(nn.Module):
    def __init__(self, tv_weight=1e-4, gp_weight=0.1):
        super().__init__()
        self.tv_loss = TotalVariationLoss(weight=tv_weight)
        self.gp_loss = GradientPenaltyLoss(weight=gp_weight)
        
    def forward(self, interpolated_seq):
        return self.tv_loss(interpolated_seq) + self.gp_loss(interpolated_seq)

