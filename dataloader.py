import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

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
        features = torch.tensor(data['features'], dtype=torch.float32)
        labels = torch.tensor(data['label'], dtype=torch.long)
        return features, labels


dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=os.cpu_count()//2,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)
