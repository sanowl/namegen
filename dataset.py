# dataset.py

import torch
from torch.utils.data import Dataset

class NameDataset(Dataset):
    def __init__(self, names, pad_token='<PAD>'):
        self.names = names
        self.pad_token = pad_token
        
        # Ensure the pad token is included in the vocabulary
        all_chars = sorted(set(''.join(names)))
        if pad_token not in all_chars:
            all_chars.append(pad_token)
        
        self.char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.max_length = max(len(name) for name in names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        input_seq = [self.char_to_idx[c] for c in name[:-1]]
        target_seq = [self.char_to_idx[c] for c in name[1:]]
        
        # Padding sequences
        padding_length = self.max_length - len(input_seq) - 1
        input_seq += [self.char_to_idx[self.pad_token]] * padding_length
        target_seq += [self.char_to_idx[self.pad_token]] * padding_length
        
        # Create a mask where 1 indicates real data and 0 indicates padding
        mask = [1] * len(name[:-1]) + [0] * padding_length
        
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float)
        )
