import random
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F


class PianoRollDataset(torch.utils.data.Dataset):
    def __init__(self, sg_files, pr_files, device, pr_max):
        self.sg_files = sg_files
        self.pr_files = pr_files
        self.pr_max = pr_max
        self.device = device

    def __len__(self):
        return len(self.sg_files)

    def __getitem__(self, idx: int) -> Dict:
        sg_path = self.sg_files[idx]
        pr_path = self.pr_files[idx]

        assert str(sg_path).rpartition("-spec.npy")[0] == str(pr_path).rpartition("-piano.npy")[0]

        sg_arr = np.load(sg_path)
        pr_arr = np.load(pr_path)
        
        sg_tensor = torch.tensor(sg_arr, requires_grad=True).to(dtype=torch.float32).unsqueeze(0)
        pr_tensor = torch.tensor(pr_arr, requires_grad=True).to(dtype=torch.float32).unsqueeze(0)
        pr_tensor /= self.pr_max
        
        return {"x": sg_tensor, "label_ids": pr_tensor}


def match_pr(sg):
    """
    retrieves the matching pr file for a given spec file path
    """
    file_pattern = sg.rpartition("\\")[2].replace("-spec.npy", "")
    return sg.replace(f"{file_pattern}-spec.npy", f"{file_pattern}-piano.npy")


def reduce_ds_size(train_sg, train_pr, val_sg, val_pr, ds_prop):
    """
    reduces dataset size by ds_prop with random selection
    """
    train_amt = int(len(train_sg) * ds_prop)
    val_amt = int(len(val_sg) * ds_prop)

    train_sg = random.sample(train_sg, k=train_amt)
    val_sg = random.sample(val_sg, k=val_amt)

    train_pr = [match_pr(x) for x in train_sg]
    val_pr = [match_pr(x) for x in val_sg]

    return train_sg, train_pr, val_sg, val_pr


def collate_fn(batch: Dict) -> Dict:
    max_pr_len = max([item['label_ids'].shape[2] for item in batch])
    max_sg_len = max([item['x'].shape[2] for item in batch])


    sg_padded = []
    pr_padded = []
    
    for item in batch:
        sg = item['x']  
        pr = item['label_ids'] 
        
        # pad spectrogram
        pad_len_sg = max_sg_len - sg.shape[2]
        sg = F.pad(sg, (0, pad_len_sg), mode="constant", value=-1)
        sg_padded.append(sg)
        
        # pad piano roll and add eos token
        pad_len_pr = max_pr_len - pr.shape[2]

        eos = torch.nn.functional.one_hot(torch.tensor(128)).unsqueeze(0).unsqueeze(-1)

        pr_with_eos = torch.cat([pr, 
                               eos.to(pr.device)], dim=2)
        pr_with_eos = F.pad(pr_with_eos, (0, pad_len_pr), mode="constant", value=-1)
        pr_padded.append(pr_with_eos)
    
    sg_batch = torch.stack(sg_padded)
    pr_batch = torch.stack(pr_padded)
    
    return {"x": sg_batch, "labels": pr_batch}


def open_csv(csv_path: str) -> List:
    csv_list = []
    with open(csv_path) as opened_csv:
        for row in opened_csv:
            # :-1 to get rid of next line character
            csv_list.append(row[:-1])
    return csv_list
