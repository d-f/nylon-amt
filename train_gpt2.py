from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config
import torch.nn as nn
from typing import Type
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from bitsandbytes.optim import Adam8bit
import bitsandbytes as bnb
import torch.nn.functional as F


class PianoGPT(nn.Module):
    def __init__(self, input_dim, pr_dim, gpt, embed_dim, sg_dim, device, max_token_gen):
        super(PianoGPT, self).__init__()
        self.input_dim = input_dim
        self.gpt = gpt
        self.sg_embed = nn.Sequential(
            nn.Linear(sg_dim, embed_dim),
            nn.LayerNorm(embed_dim), 
            nn.Dropout(0.2)
        )
        self.pr_embed = nn.Sequential(
            nn.Linear(pr_dim, embed_dim),
            nn.LayerNorm(embed_dim),  
            nn.Dropout(0.2)
        )
        self.embed_dim = embed_dim
        self.device = device
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.gpt.to(device)
        self.pr_embed.to(device)
        self.max_token_gen = max_token_gen

        self.eos_token = torch.zeros(pr_dim).to(device)
        self.eos_token[-1] = 1  

    def embed_pr(self, x):
        return self.pr_embed(x)
    
    def embed_sg(self, x):
        x = x.squeeze(1) 
        x = x.transpose(1, 2)
        embedded = self.sg_embed(x)
        return embedded
    
    def forward(self, x, labels=None):
        batch_size = x.shape[0]

        sg_embedded = self.embed_sg(x)
        
        input_tokens = sg_embedded
        
        outputs = []
        total_loss = 0.0
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool).to(self.device)

        if labels is not None:
            seq_length = labels.size(3)
        else:
            seq_length = self.max_token_gen

        for t in range(seq_length):
            if input_tokens.size(1) > self.gpt.config.n_positions:
                input_tokens = input_tokens[:, -self.gpt.config.n_positions:, :]

            gpt_output = self.gpt(inputs_embeds=input_tokens).logits
            next_token_logits = gpt_output[:, -1, :]
            
            outputs.append(next_token_logits)

            # inference
            if labels is None:
                next_token_preds = (torch.sigmoid(next_token_logits) > 0.5).float()
                eos_detected = torch.all(torch.abs(next_token_preds - self.eos_token) < 1e-6, dim=1)
                eos_ground_truth = torch.all(torch.abs(labels[:, t] - self.eos_token) < 1e-6, dim=1)
                eos_detected = eos_detected & eos_ground_truth 
                finished_sequences = finished_sequences | eos_detected
                if torch.all(finished_sequences):
                    break

                next_token = next_token_preds
            else:
                # teacher forcing for training
                next_token = labels.squeeze(1)[:, :, t]
                step_loss = self.loss_fn(next_token_logits, next_token)
                total_loss += step_loss

            next_token_embed = self.embed_pr(next_token).unsqueeze(1)  
            input_tokens = torch.cat([input_tokens, next_token_embed], dim=1)

        outputs = torch.stack(outputs, dim=1)  

        if labels is not None:
            return total_loss / seq_length, outputs
        return outputs


class PianoDataset(torch.utils.data.Dataset):
    def __init__(self, sg_files, pr_files, device, pr_max):
        self.sg_files = sg_files
        self.pr_files = pr_files
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
        
        sg_tensor = torch.tensor(sg_arr, requires_grad=True).to(dtype=torch.float32).unsqueeze(0)
        pr_tensor = torch.tensor(pr_arr, requires_grad=True).to(dtype=torch.float32).unsqueeze(0)
        pr_tensor /= self.pr_max
        
        return {"x": sg_tensor, "label_ids": pr_tensor}


def define_model(spec_len: int, pr_dim: int, embed_dim, sg_dim, device, max_gen) -> Type[GPT2LMHeadModel]:
    """
    returns GPT2 model
    """
    config = GPT2Config.from_pretrained('gpt2')
    config.n_inner = 256
    config.n_layer = 2
    config.n_head = 2
    config.n_positions = 512
    config.n_embd = embed_dim
    gpt_model = GPT2LMHeadModel(config)
    gpt_model.resize_token_embeddings(new_num_tokens=pr_dim)
    gpt_model.gradient_checkpointing_enable()
    piano_gpt = PianoGPT(input_dim=spec_len, pr_dim=pr_dim, gpt=gpt_model, embed_dim=embed_dim, sg_dim=sg_dim, device=device, max_token_gen=max_gen)
    
    return piano_gpt


def collate_fn(batch):
    max_sg_len = max([item['x'].shape[2] for item in batch])
    max_pr_len = max([item['label_ids'].shape[2] for item in batch])

    sg_padded = []
    pr_padded = []
    
    for item in batch:
        sg = item['x']  
        pr = item['label_ids'] 
        
        # pad spectrogram
        pad_len_sg = max_sg_len - sg.shape[2]
        sg_padded.append(F.pad(sg, (0, pad_len_sg)))
        
        # pad piano roll and add eos token
        pad_len_pr = max_pr_len - pr.shape[2]
        pr_with_eos = torch.cat([pr, 
                                torch.zeros(*pr.shape[:2], 1).to(pr.device)], dim=2)
        pr_with_eos = F.pad(pr_with_eos, (0, pad_len_pr))
        pr_padded.append(pr_with_eos)
    
    sg_batch = torch.stack(sg_padded)
    pr_batch = torch.stack(pr_padded)
    
    return {"x": sg_batch, "labels": pr_batch}


def open_csv(csv_path):
    csv_list = []
    with open(csv_path) as opened_csv:
        for row in opened_csv:
            # :-1 to get rid of next line character
            csv_list.append(row[:-1])
    return csv_list


def test_model(model, test_ds, device):
    pred_list = []
    label_list = []

    test_dl = DataLoader(test_ds, batch_size=64, collate_fn=collate_fn)

    for output_dict in tqdm(test_dl):
        pred = model(output_dict['x'].to(device)).transpose(1, 2)
        labels = output_dict["labels"].to(device)
        pred = (pred > 0.5)*1
        pred_list.append(pred)
        label_list.append(labels)

    
    predictions = torch.cat(pred_list, dim=0)
    labels = torch.cat(label_list, dim=0)

    predicted_notes = (torch.sigmoid(predictions) > 0.5).float()
    accuracy = (predicted_notes == labels.squeeze(1)).float().mean()

    return accuracy


def train(num_epochs):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda")

    train_sg = open_csv("C:\\personal_ML\\music-transcription\\train_sg.csv")
    train_pr = open_csv("C:\\personal_ML\\music-transcription\\train_pr.csv")
    val_sg = open_csv("C:\\personal_ML\\music-transcription\\val_sg.csv")
    val_pr = open_csv("C:\\personal_ML\\music-transcription\\val_pr.csv")
    test_sg = open_csv("C:\\personal_ML\\music-transcription\\test_sg.csv")
    test_pr = open_csv("C:\\personal_ML\\music-transcription\\test_pr.csv")

    train_ds = PianoDataset(device=device, pr_max=204, sg_files=train_sg, pr_files=train_pr)
    val_ds = PianoDataset(device=device, pr_files=val_pr, sg_files=val_sg, pr_max=204)
    test_ds = PianoDataset(device=device, pr_files=test_pr, sg_files=test_sg, pr_max=204)
    model = define_model(spec_len=100+1, pr_dim=128+1, embed_dim=128, sg_dim=40, device=device, max_gen=256).to(device)

    lr = 1e-4
    batch_size = 64
    acc = test_model(model=model, test_ds=test_ds, device=device)
    print(acc)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir="C:\\personal_ML\\music-transcription\\logs",
        learning_rate=lr,
        optim="adamw_hf",
        logging_steps=1,
        max_grad_norm=1.0,
        do_eval=True,
        eval_steps=int(len(train_sg) / batch_size),
        eval_strategy="steps",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
        eval_dataset=val_ds
    )

    trainer.train()
    torch.save(model, "C:\\personal_ML\\music-transcription\\model_1.pth.tar")    
    acc = test_model(model=model, test_ds=test_ds, device=device)
    print(acc)


def main():
    train(num_epochs=1)

if __name__ == "__main__":
    main()
