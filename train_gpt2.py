from pathlib import Path
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config, AdamW
import torch.nn as nn
from typing import Type
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel, GPT2Config, AdamW, Trainer, TrainingArguments


class PianoGPT(nn.Module):
    def __init__(self, input_dim, pr_dim, gpt, embed_dim, sg_dim, device):
        super(PianoGPT, self).__init__()
        self.input_dim = input_dim
        self.gpt = gpt
        self.sg_embed = nn.Sequential(
    nn.Linear(sg_dim, embed_dim),
    nn.LayerNorm(embed_dim), # Normalize after embedding
    nn.Dropout(0.1)
)
        self.pr_embed = nn.Sequential(
    nn.Linear(pr_dim, embed_dim),
    nn.LayerNorm(embed_dim),  # Normalize after embedding
    nn.Dropout(0.1)
)
        self.embed_dim = embed_dim
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        sg_embedded = self.embed_sg(x)  # Embed the spectrogram (shape: batch_size, seq_len, embed_dim)
        # pr_embedded = torch.zeros_like(sg_embedded).to(self.device)  # Initialize piano roll output (all zeros)

        # Start with an SOS token or the first piano roll token
        sos_token = torch.zeros((x.shape[0], 1, self.embed_dim)).to(self.device)  # SOS token (batch_size, 1, embed_dim)
        input_tokens = torch.cat([sos_token, sg_embedded], dim=1)  # Combine SOS token and spectrogram
        
        outputs = []
        loss = 0
        self.gpt.to(self.device)
        self.pr_embed.to(self.device)

        for t in range(1, sg_embedded.size(1) + 1):  # Auto-regressively generate tokens
            gpt_output = self.gpt(inputs_embeds=input_tokens).logits  # Get the next token logits from GPT-2
            
            next_token_logits = gpt_output[:, -1, :]  # Only take the last token's logits
            
            outputs.append(next_token_logits.unsqueeze(1))  # Append the predicted token for output
            
            # Update input tokens for the next step
            next_token_embed = self.embed_pr(next_token_logits.float())  # Embed the predicted token
            input_tokens = torch.cat([input_tokens[:, 1:, :], next_token_embed.unsqueeze(0)], dim=1)  # Concatenate to input tokens

            # Compute loss if labels are provided
            if labels is not None:
                target_token = labels[:, :, :, t].squeeze(0).to(self.device)  # Get the ground truth token
                loss += self.loss_fn(next_token_logits, target_token)  # Calculate cross-entropy loss
            
        # Stack the outputs into the final predicted piano roll
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
    config.n_positions = spec_len
    config.n_inner = int(768 / 2)
    config.n_layer = 2
    config.n_heads = 2
    gpt_model = GPT2LMHeadModel(config)
    gpt_model.resize_token_embeddings(new_num_tokens=pr_dim)
    gpt_model.gradient_checkpointing_enable()
    piano_gpt = PianoGPT(input_dim=spec_len, pr_dim=pr_dim, gpt=gpt_model, embed_dim=embed_dim, sg_dim=sg_dim, device=device)
    
    return piano_gpt


def train(num_epochs):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda")
    dataset = PianoDataset(device=device, pr_max=204, file_dir=Path("C:\\personal_ML\\music-transcription\\save\\"))
    model = define_model(spec_len=100+1, pr_dim=128, embed_dim=768, sg_dim=40, device=device).to(device)
    for param in model.parameters():
        param.requires_grad = True

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        num_train_epochs=10,
        logging_dir="C:\\personal_ML\\music-transcription\\logs",
        learning_rate=1e-3,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()


def main():
    train(num_epochs=10)


if __name__ == "__main__":
    main()
