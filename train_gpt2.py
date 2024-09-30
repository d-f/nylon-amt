import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config, AdamW
import torch.nn as nn
from typing import Type


def define_model(spec_len: int, pr_dim: int) -> Type[GPT2LMHeadModel]:
    """
    returns GPT2 model
    """
    config = GPT2Config.from_pretrained('gpt2')
    config.n_positions = spec_len
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(new_num_tokens=pr_dim)
    return model


def define_sg_embedding(sg_dim, embed_dim):
    """
    returns the embedding model to transform
    the spectrogram vector into the correct shape
    for gpt2 input
    """
    return nn.Linear(sg_dim, embed_dim)


def define_pr_embedding(pr_dim, embed_dim):
    """
    returns the embedding model to transform
    the spectrogram vector into the correct shape
    for gpt2 input
    """
    return nn.Linear(pr_dim, embed_dim)


def train(num_epochs):
    sg_arr = np.load("C:\\personal_ML\\music-transcription\\save\\0-5000-MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav-spec.npy")
    pr_arr = np.load("C:\\personal_ML\\music-transcription\\save\\5000-10000-MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav-piano.npy")

    device = torch.device("cuda")

    sg_tensor = torch.tensor(sg_arr).to(device).unsqueeze(0)
    pr_tensor = torch.tensor(pr_arr).to(device, dtype=torch.float32).unsqueeze(0)

    model = define_model(spec_len=5001, pr_dim=128).to(device)
    sg_embedding = define_sg_embedding(sg_dim=40, embed_dim=model.config.n_embd).to(device)
    pr_embedding = define_pr_embedding(pr_dim=128, embed_dim=model.config.n_embd).to(device)
    optimizer = AdamW(list(pr_embedding.parameters()) + list(sg_embedding.parameters()) + list(model.parameters()), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    embedded_input = sg_embedding(sg_tensor.transpose(1, 2))
    sos_token = torch.zeros(size=(1, model.config.n_embd)).to(device).unsqueeze(0)

    embedded_input = torch.cat(tensors=[embedded_input, sos_token], dim=1)

    # transpose shape to (128, n) -> (n, 128) so that the vector can be
    # interpreted as a sequence of 128-dim piano roll vectors rather than
    # a separate sequence for each piano roll note
    pr_embed = pr_embedding(pr_tensor.transpose(1, 2))
    eos_token = torch.zeros(size=(1, model.config.n_embd)).to(device).unsqueeze(0)

    pr_embed = torch.cat(tensors=[pr_embed, eos_token], dim=1)

    for epoch_idx in range(num_epochs):
        for pr_idx in range(pr_embed.shape[1]):
            output = model(inputs_embeds=embedded_input)
            next_token = output.logits[:, -1, :]
            loss = criterion(next_token, pr_tensor[:, :, pr_idx])
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # shift input and concatenate ground truth piano roll 
            # for teacher forcing
            embedded_input = torch.cat(tensors=[embedded_input[:, 1:, :], pr_embed[:, pr_idx, :].unsqueeze(0)], dim=1)


def main():
    train(num_epochs=10)


if __name__ == "__main__":
    main()
