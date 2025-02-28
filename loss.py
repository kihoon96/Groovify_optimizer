
import torch
import torch.nn as nn
from torch.fft import rfft, rfftfreq
from torch.nn import functional as F

# from pytorch_metric_learning.losses import NTXentLoss

from vis import SMPLSkeleton

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, \
                                 matrix_to_rotation_6d, rotation_6d_to_matrix

smpl = SMPLSkeleton('cuda')

def ax_from_6v(q):
    """ Converts a 6D rotation tensor back to an axis-angle representation. """
    assert q.shape[-1] == 6, "Input tensor must be of shape (*, 6)"
    return matrix_to_axis_angle(rotation_6d_to_matrix(q))

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

class RhythmLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.energy_fn = y_energy_batch
        self.smpl = smpl
        
    def forward(self, signal_batch, bpm_batch):
        b, s, c = signal_batch.shape
        model_x = signal_batch[:, :, :3]
        model_q = ax_from_6v(signal_batch[:, :, 3:].reshape(b, s, -1, 6))        
        
        model_xp = self.smpl.forward(model_q, model_x)
        model_xp_v = model_xp[:, 1:] - model_xp[:, :-1]
        model_xp_a = model_xp_v[:, 1:] - model_xp_v[:, :-1]

        model_xp_a_norm = torch.sqrt(torch.sum(model_xp_a**2, axis=3))

        N = 9* model_xp.shape[1]
        #case1 vel - adoption nowon
        # model_out_padded = F.pad(model_out_a_fft, (0,0,0,N), mode='constant')
        # #case2 accel
        # model_out_padded = F.pad(model_out_v_fft, (0,0,0,N), mode='constant')
        #case3 forward
        model_out_padded = F.pad(model_xp_a_norm, (0,0,0,N), mode='constant')

        rfft_model_out = torch.fft.rfft(model_out_padded, dim=1)
        out_magnitude = torch.sqrt(rfft_model_out.real**2 + rfft_model_out.imag**2 + 1e-10)
        
        bq_batch = bpm_batch/60.0
        rhythm_energy = y_energy_batch(out_magnitude.sum(dim=2), bq_batch)
        fft_loss = rhythm_energy.mean() # / 10000.0

        return fft_loss
    
    
    
# class RhythmLoss(nn.Module):
#     def __init__(self, base_freq=1.0, num_bands=8, temp=0.1):
#         super().__init__()
#         self.freq_weights = nn.Parameter(torch.ones(num_bands))
#         self.base_freq = base_freq
#         self.temp = temp
#         # self.contrastive = NTXentLoss(temperature=temp)  # Contrastive regularization
        
#     def forward(self, fftsignal_batch, bq_batch, motion_features):
#         # Spectral attention with learnable weights
#         rhythm_energy = self.spectral_attention(fftsignal_batch, bq_batch)
        
#         # Contrastive regularization in motion feature space
#         # contrast_loss = self.contrastive(motion_features)
        
#         # Temporal consistency from DTW alignment [5][7]
#         dtw_loss = self.dtw_alignment_loss(fftsignal_batch)
        
#         return rhythm_energy # +  0.3*contrast_loss + 0.2*dtw_loss

#     def spectral_attention(self, fftsignal, bq_tensor):
#         # Multi-band frequency masking with learnable importance
#         bq_factors = torch.stack([bq_tensor * (2**i) for i in range(4)], dim=1)
#         freq_bins = self._create_adaptive_bins(bq_factors)
        
#         # Focal frequency weighting [11]
#         band_energies = []
#         for i, (low, high) in enumerate(freq_bins):
#             mask = (self.frequencies >= low) & (self.frequencies <= high)
#             band_energy = torch.mean(torch.abs(fftsignal[:,mask]) * self.freq_weights[i])
#             band_energies.append(band_energy)
            
#         return sum(band_energies) / len(band_energies)

#     def dtw_alignment_loss(self, fft_features):
#         # Differentiable DTW alignment [5][7]
#         alignment_cost = 0
#         for i in range(fft_features.shape[0]-1):
#             alignment_cost += torch.mean(
#                 SoftDTW(fft_features[i], fft_features[i+1], gamma=1.0)
#             )
#         return alignment_cost

# class FocalFrequencyLoss(nn.Module):  # From [11]
#     def __init__(self, loss_weight=1.0, alpha=1.0):
#         super().__init__()
#         self.loss_weight = loss_weight
#         self.alpha = alpha

#     def forward(self, pred, target):
#         pred_fft = torch.fft.fft2(pred)
#         target_fft = torch.fft.fft2(target)
        
#         freq_weight = self._get_freq_weight(pred_fft, target_fft)
#         loss = torch.mean(freq_weight * torch.abs(pred_fft - target_fft)**2)
#         return self.loss_weight * loss

# class MotionRegularizer(nn.Module):
#     def __init__(self, tv_weight=1e-4, gp_weight=0.1):
#         super().__init__()
#         self.tv_loss = TotalVariationLoss(weight=tv_weight)
#         self.gp_loss = GradientPenaltyLoss(weight=gp_weight)
        
#     def forward(self, interpolated_seq):
#         return self.tv_loss(interpolated_seq) + self.gp_loss(interpolated_seq)

