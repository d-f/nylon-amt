import argparse
from pathlib import Path
import torchaudio
import torch
import pickle
from hftt_code.corpus.conv_midi2note import midi2note
from hftt_code.corpus.conv_note2label import note2label
from tqdm import tqdm


def parse_cla():
    parser = argparse.ArgumentParser()
    parser.add_argument("-maestro_dir", type=Path)
    parser.add_argument("-save_dir", type=Path)
    return parser.parse_args()


def process_audio(fp, save_fp, tr_mel):
    wave, sr = torchaudio.load(fp)
    wave_mono = torch.mean(wave, dim=0)
    resample_transform = torchaudio.transforms.Resample(sr, 16e3)
    wave_mono_16k = resample_transform(wave_mono)    
    mel_spec = tr_mel(wave_mono_16k)
    feature = (torch.log(mel_spec + 1e-8)).T
    save_pickle(save_fp, pickle_obj=feature)


def process_midi(midi_fp, save_fp):
    notes = midi2note(midi_fp)
    labels = note2label(f_note=notes, offset_duration_tolerance_flag=False)
    save_pickle(save_fp=save_fp, pickle_obj=labels)


def pattern_search(dir, pattern):
    return [x for x in dir.glob(f"*{pattern}*")]


def save_pickle(save_fp, pickle_obj):
    with open(save_fp, 'wb') as opened_pickle:
        pickle.dump(pickle_obj, opened_pickle, protocol=4)


def main():
    args = parse_cla()
    tr_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16e3, 
        n_fft=2048, 
        win_length=2048, 
        hop_length=256, 
        pad_mode="constant", 
        n_mels=256, 
        norm='slaney'
        )
    for potential_folder in tqdm(args.maestro_dir.iterdir(), desc="Folder"):
        if args.maestro_dir.joinpath(potential_folder).is_dir():
            audio_files = pattern_search(
                dir=args.maestro_dir.joinpath(potential_folder), pattern=".wav"
            )
            midi_files = pattern_search(
                dir=args.maestro_dir.joinpath(potential_folder), pattern=".midi"
            )
            for audio_fp in tqdm(audio_files, desc="Audio"):
                process_audio(
                    fp=audio_fp, 
                    save_fp=args.save_dir.joinpath(audio_fp.stem+"_feature.pkl"), 
                    tr_mel=tr_mel
                    )
            for midi_fp in tqdm(midi_files, desc="Midi"):
                process_midi(
                    midi_fp=midi_fp,
                    save_fp=args.save_dir.joinpath(midi_fp.stem+"_label.pkl")
                )
            

if __name__ == "__main__":
    main()
