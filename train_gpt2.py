import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config, AdamW
import torch.nn as nn
from typing import Type
from tqdm import tqdm


class PianoGPT(nn.Module):
    def __init__(self, input_dim, pr_dim, gpt, embed_dim, sg_dim, device):
        super(PianoGPT, self).__init__()
        self.input_dim = input_dim
        self.gpt = gpt
        self.sg_embed = nn.Linear(sg_dim, embed_dim)
        self.pr_embed = nn.Linear(pr_dim, embed_dim)
        self.embed_dim = embed_dim
        self.device = device

    def forward(self, x):
        out = self.gpt(inputs_embeds=x)
        next_token = out.logits[:, -1, :]
        return next_token

    def embed_pr(self, x):
        pr_embed = self.pr_embed(x.transpose(1, 2))
        eos_token = torch.zeros(size=(1, self.embed_dim)).unsqueeze(0).to(self.device)
        pr_embed = torch.cat(tensors=[pr_embed, eos_token], dim=1)
        return pr_embed
    
    def embed_sg(self, x):
        embedded_input = self.sg_embed(x.transpose(1, 2))
        sos_token = torch.zeros(size=(1, self.embed_dim)).unsqueeze(0).to(self.device)
        embedded_input = torch.cat(tensors=[embedded_input, sos_token], dim=1)
        return embedded_input


def define_model(spec_len: int, pr_dim: int, embed_dim, sg_dim, device) -> Type[GPT2LMHeadModel]:
    """
    returns GPT2 model
    """
    config = GPT2Config.from_pretrained('gpt2')
    config.n_positions = spec_len
    gpt_model = GPT2LMHeadModel(config)
    gpt_model.resize_token_embeddings(new_num_tokens=pr_dim)
    gpt_model.gradient_checkpointing_enable()
    piano_gpt = PianoGPT(input_dim=spec_len, pr_dim=pr_dim, gpt=gpt_model, embed_dim=embed_dim, sg_dim=sg_dim, device=device)
    
    return piano_gpt


def train(num_epochs):
    sg_arr = np.load("C:\\personal_ML\\music-transcription\\save\\0-5000-MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav-spec.npy")
    pr_arr = np.load("C:\\personal_ML\\music-transcription\\save\\5000-10000-MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav-piano.npy")

    device = torch.device("cuda")

    sg_tensor = torch.tensor(sg_arr).to(device).unsqueeze(0)
    pr_tensor = torch.tensor(pr_arr).to(device, dtype=torch.float32).unsqueeze(0)

    model = define_model(spec_len=5001, pr_dim=128, embed_dim=768, sg_dim=40, device=device).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch_idx in tqdm(range(num_epochs), desc="Epoch"):
        piano = model.embed_pr(pr_tensor)
        embedded_input = model.embed_sg(sg_tensor)
        pr_embed_slices = torch.split(piano, split_size_or_sections=1, dim=1)
        pr_tensor_slices = torch.split(pr_tensor, split_size_or_sections=1, dim=2)

        for pr_idx in tqdm(range(len(pr_embed_slices)), desc="Piano Roll"):
            next_token = model(embedded_input)
            loss = criterion(next_token, pr_tensor_slices[pr_idx].squeeze(-1))
            print(loss)
            optimizer.zero_grad()
            if pr_idx < pr_tensor.shape[2] - 2:
                loss.backward(retain_graph=True)  # Retain graph for future backpropagation
            else:
                loss.backward() 
            optimizer.step()

            # shift input and concatenate ground truth piano roll 
            # for teacher forcing
            embedded_input = torch.cat(tensors=[embedded_input[:, 1:, :], pr_embed_slices[pr_idx]], dim=1)        


def main():
    train(num_epochs=10)


if __name__ == "__main__":
    main()
