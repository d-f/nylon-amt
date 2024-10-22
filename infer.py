import argparse
from pathlib import Path
import torch
from typing import Dict
from dataset.dataset_creation import load_wav, audio_to_mel, convert_to_DB, norm_sg
from test import load_model
from train import define_model
import math
from torch.utils.data import DataLoader


def parse_cla():
    parser = argparse.ArgumentParser()
    # path to the audio to transcribe
    parser.add_argument("-audio_path", type=Path, default="C:\\personal_ML\\nylon_gpt\\raw_data\\maestro-v3.0.0\\2004\\MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav")
    # sample rate of the audio
    parser.add_argument("-sr", type=int, default=16000)
    # length of the spectrogram
    parser.add_argument("-chunk_len", type=int, default=100)
    # path to the model checkpoint
    parser.add_argument("-chk", type=str, default="C:\\personal_ML\\nylon_gpt\\training_results\\model_3\\checkpoint-2298\\pytorch_model.bin")
    # length of the audio segments
    parser.add_argument("-spec_len", type=int, default=100+1) 
    # number of features for piano roll (+1 for eos)
    parser.add_argument("-pr_dim", type=int, default=129)
    # number of spectrogram features 
    parser.add_argument("-sg_dim", type=int, default=40) 
    # size of the embedding dimension
    parser.add_argument("-embed_dim", type=int, default=200) 
    # dimensionality of feed forward layers in transformer
    parser.add_argument("-n_inner", type=int, default=512) 
    # number of hidden layers in the transformer
    parser.add_argument("-n_layer", type=int, default=4)
    # number of attention heads 
    parser.add_argument("-n_head", type=int, default=4) 
    # maximum sequence length
    parser.add_argument("-n_positions", type=int, default=150) 
    # path to save predicted piano roll to
    parser.add_argument("-save_dir", type=Path, default=Path("C:\\personal_ML\\nylon_gpt\\inference_results\\"))
    return parser.parse_args()


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, audio_path, sr, chunk_len):
        self.audio, _ = load_wav(file_path=audio_path, sample_rate=sr)
        mel_spectrogram = audio_to_mel(audio=self.audio, sr=sr, n_mels=40)
        mel_spectrogram_db = convert_to_DB(
        mel_spectrogram=mel_spectrogram,
        multiplier=10, 
        amin=1e-10,
        db_multiplier=0.0,
        top_db=80.0
        )
        self.norm_db = norm_sg(mel_spectrogram_db)
        self.chunk_len = chunk_len
        self.pos_list = [x for x in range(0, self.norm_db.shape[1], self.chunk_len)]

    def __len__(self):
        song_len = self.norm_db.shape[1]
        return math.ceil(song_len / self.chunk_len)

    def __getitem__(self, idx: int) -> Dict:
        pos = self.pos_list[idx]
        start = pos*self.chunk_len
        stop = start + self.chunk_len
        sg_chunk = self.norm_db[:, start:stop].numpy() 
        sg_tensor = torch.tensor(sg_chunk, requires_grad=True).to(dtype=torch.float32).unsqueeze(0)
        return sg_tensor
    

def save_pred(batch_pred, save_dir, audio_path):
    for pred_idx in range(batch_pred.shape[1]):
        pred = batch_pred[:, pred_idx, :]
        file_name = f"{audio_path.name}-pred.pt"
        torch.save(obj=pred, f=save_dir.joinpath(file_name))
    

def infer(inf_dataloader, model, save_dir, device, chunk_len, audio_path):
    for batch_tensor in inf_dataloader:
        batch_pred = model(batch_tensor.to(device))
        save_pred(batch_pred, save_dir, audio_path)


def main():
    args = parse_cla()
    device = torch.device("cuda")
    inf_ds = InferenceDataset(
        audio_path=args.audio_path, sr=args.sr, chunk_len=args.chunk_len
    )
    inf_dl = DataLoader(dataset=inf_ds, shuffle=False, batch_size=2)
    model = define_model(
        spec_len=args.spec_len, 
        pr_dim=args.pr_dim, 
        embed_dim=args.embed_dim, 
        sg_dim=args.sg_dim, 
        device=device, 
        max_gen=args.n_positions,
        n_inner=args.n_inner,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_positions=args.n_positions
        ).to(device)
    load_model(model=model, checkpoint_path=args.chk)
    infer(inf_dataloader=inf_dl, model=model, save_dir=args.save_dir, device=device, chunk_len=args.chunk_len, audio_path=args.audio_path)


if __name__ == "__main__":
    main()
