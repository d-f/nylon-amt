import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def define_model():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(new_num_tokens=128)
    return model


def main():
    sg_arr = np.load("C:\\personal_ML\\music-transcription\\save\\0-5000-MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav-spec.npy")
    pr_arr = np.load("C:\\personal_ML\\music-transcription\\save\\5000-10000-MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav-piano.npy")

    device = torch.device("gpu")

    sg_tensor = torch.tensor(sg_arr).to(device)
    pr_tensor = torch.tensor(pr_arr).to(device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = define_model().to(device)


if __name__ == "__main__":
    main()
  
