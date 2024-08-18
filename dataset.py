# dataset.py

import torch
from torch.utils.data import Dataset

class NameDataset(Dataset):
    def __init__(self, names):
        self.names = names
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(set(''.join(names))))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        input_seq = [self.char_to_idx[c] for c in name[:-1]]
        target_seq = [self.char_to_idx[c] for c in name[1:]]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
