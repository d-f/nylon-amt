from pathlib import Path
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config
import torch.nn as nn
from typing import Type
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
import bitsandbytes as bnb
from bitsandbytes.optim import Adam8bit


class PianoGPT(nn.Module):
    def __init__(self, input_dim, pr_dim, gpt, embed_dim, sg_dim, device):
        super(PianoGPT, self).__init__()
        self.input_dim = input_dim
        self.gpt = gpt
        self.sg_embed = nn.Sequential(
                            nn.Linear(sg_dim, embed_dim),
                            nn.LayerNorm(embed_dim), 
                            nn.Dropout(0.1)
                        )
        self.pr_embed = nn.Sequential(
                            nn.Linear(pr_dim, embed_dim),
                            nn.LayerNorm(embed_dim),  
                            nn.Dropout(0.1)
                        )
        self.embed_dim = embed_dim
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        sg_embedded = self.embed_sg(x)  

        sos_token = torch.zeros((x.shape[0], 1, self.embed_dim)).to(self.device)  
        input_tokens = torch.cat([sos_token, sg_embedded], dim=1)  
        
        outputs = []
        loss = 0
        self.gpt.to(self.device)
        self.pr_embed.to(self.device)

        for t in range(1, sg_embedded.size(1) + 1):  
            gpt_output = self.gpt(inputs_embeds=input_tokens).logits 
            
            next_token_logits = gpt_output[:, -1, :]  
            
            outputs.append(next_token_logits.unsqueeze(1))  
        
            next_token_embed = self.embed_pr(next_token_logits.float()) 
            input_tokens = torch.cat([input_tokens[:, 1:, :], next_token_embed.unsqueeze(1)], dim=1) 

            if labels is not None:
                target_token = labels[:, :, :, t].squeeze(0).to(self.device) 
                loss += self.loss_fn(next_token_logits, target_token.squeeze(1))  
            
        outputs = torch.cat(outputs, dim=1)
        
        return (loss, outputs) if labels is not None else outputs

    def embed_pr(self, x):
        pr_embed = self.pr_embed(x)
        return pr_embed
    
    def embed_sg(self, x):
        x = x.squeeze(1)  
        x = x.transpose(1, 2)        
        embedded_input = self.sg_embed(x).to(self.device)
        return embedded_input
    

class PianoDataset(torch.utils.data.Dataset):
    def __init__(self, file_dir, device, pr_max):
        self.sg_files = [x for x in file_dir.iterdir() if "spec" in str(x)]
        self.pr_files = [x for x in file_dir.iterdir() if "piano" in str(x)]
        self.pr_max = pr_max
        self.device = device

    def __len__(self):
        return len(self.sg_files)

    def __getitem__(self, idx):
        sg_path = self.sg_files[idx]
        pr_path = self.pr_files[idx]

        assert str(sg_path).rpartition("-spec.npy")[0] == str(pr_path).rpartition("-piano.npy")[0]

        sg_arr = np.load(sg_path)
        pr_arr = np.load(pr_path)
        
        sg_tensor = torch.tensor(sg_arr, requires_grad=True).unsqueeze(0)
        pr_tensor = torch.tensor(pr_arr, requires_grad=True).to(dtype=torch.float32).unsqueeze(0)
        pr_tensor /= self.pr_max
        
        return {"x": sg_tensor, "label_ids": pr_tensor}


def define_model(spec_len: int, pr_dim: int, embed_dim, sg_dim, device) -> Type[GPT2LMHeadModel]:
    """
    returns GPT2 model
    """
    config = GPT2Config.from_pretrained('gpt2')
    config.n_inner = int(768 / 5)
    config.n_layer = 1
    config.n_heads = 1
    config.n_positions = 512
    gpt_model = GPT2LMHeadModel(config)
    gpt_model.resize_token_embeddings(new_num_tokens=pr_dim)
    gpt_model.gradient_checkpointing_enable()
    for param in gpt_model.parameters():
        param.data = bnb.nn.Int8Params(param.data, requires_grad=True)
    piano_gpt = PianoGPT(input_dim=spec_len, pr_dim=pr_dim, gpt=gpt_model, embed_dim=embed_dim, sg_dim=sg_dim, device=device)
    
    return piano_gpt


def collate_fn(batch):
    max_sg_len = max([item['x'].shape[2] for item in batch])  
    max_pr_len = max([item['label_ids'].shape[2] for item in batch])  

    sg_padded = []
    pr_padded = []
    
    for item in batch:
        sg = item['x']
        pr = item['label_ids']
        
        pad_len_sg = max_sg_len - sg.shape[2]
        sg_padded.append(torch.nn.functional.pad(sg, (0, pad_len_sg)))
        
        pad_len_pr = max_pr_len - pr.shape[2]
        pr_padded.append(torch.nn.functional.pad(pr, (0, pad_len_pr)))
    
    sg_batch = torch.stack(sg_padded)
    pr_batch = torch.stack(pr_padded)
    
    return {"x": sg_batch, "labels": pr_batch}


def train(num_epochs):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda")
    dataset = PianoDataset(device=device, pr_max=204, file_dir=Path("C:\\personal_ML\\music-transcription\\save\\"))
    model = define_model(spec_len=100+1, pr_dim=128, embed_dim=768, sg_dim=40, device=device).to(device)
    for param in model.parameters():
        param.requires_grad = True

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=70,
        num_train_epochs=num_epochs,
        logging_dir="C:\\personal_ML\\music-transcription\\logs",
        learning_rate=1e-3,
        logging_steps=1,
        max_grad_norm=1.0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        optimizers=(Adam8bit(model.parameters(), lr=1e-3), None)
    )

    trainer.train()


def main():
    train(num_epochs=10)


if __name__ == "__main__":
    main()
