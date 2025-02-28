import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

import re

from vis import skeleton_render_simple, SMPLSkeleton
smpl = SMPLSkeleton()

def get_gt_BPM_aistpp(wav_name):
    # 10 genres
    GT_BPMS = [
    # mBR, mPO, mLO, mMH, mLH, mWA, mKR, mJS, mJB
    [80, 90, 100, 110, 120, 130],
    # [130, 80, 90, 100, 110, 120],
    # mHO
    [110, 115, 120, 125, 130, 135]
    # [135, 110, 115, 120, 125, 130]
    ]
    music_genres = ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB', 'mHO']

    number = -1
    for target_substring in music_genres:
        match = re.search(f"{target_substring}(\d+)", wav_name)
        if match:
            number = match.group(1)
            number = int(number)
            if target_substring == 'mHO':
                target_bpm = GT_BPMS[1][number]
            else:
                target_bpm = GT_BPMS[0][number]
            return target_bpm
    assert number == -1, "No number found after the target substrings."
    return -1

class PKLDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = sorted([
            f for f in os.listdir(root_dir) 
            if f.endswith('.pkl')
        ])
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        pklname = str(file_path).split('/')[-1]
        gt_bpm = torch.tensor(get_gt_BPM_aistpp(pklname), dtype=torch.float32)
        # joint_rot = data['smpl_poses'].reshape(-1,24,3)
        joint_rot6d = torch.tensor(data['smpl_poses_6d'], dtype=torch.float32)
        joint_trans = torch.tensor(data['smpl_trans'], dtype=torch.float32)
        # joint_3d = torch.tensor(data['full_pose'], dtype=torch.float32)
        joint_input = torch.cat((joint_trans, joint_rot6d), dim=1)
        
        return gt_bpm, joint_input


if __name__  == "__main__":
    dataset = PKLDataset("./fullbaseline_split")
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=os.cpu_count()//2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    dataset.__getitem__(0)





