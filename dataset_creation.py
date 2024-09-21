from tqdm import tqdm
import pretty_midi
import matplotlib.pyplot as plt
import torch
import torchaudio
import librosa
import librosa
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np


def parse_cla() -> None:
    """
    parses command line arguments
    """
    parser = argparse.ArgumentParser()
    # folder that contains a folder for every competition year with midi and wav mixed within the same folder
    parser.add_argument("-maestro_dir", type=Path)
    # sample rate to load audio
    parser.add_argument("-sr", type=int, default=16000)
    # length of time that each piano roll and spectrogram segment should represent
    parser.add_argument("-chunk_len", type=int, default=10000) # in milliseconds
    # path to save dataset arrays to
    parser.add_argument("-ds_save_path", type=Path)
    return parser.parse_args()


def load_wav(file_path: str, sample_rate: int) -> tuple[np.array, int]:
    """
    loads wav file at a given sample rate and returns a numpy array
    of the loaded audio

    file_path   -- path to the wav file
    sample_rate -- sample rate at which the file should be loaded
    """
    audio = librosa.load(file_path, mono=True, sr=sample_rate)
    return audio


def plot_spectrogram(spectrogram: np.array) -> None:
    """
    plots spectrogram data

    spectrogram -- np.array of spectrogram values
    """
    plt.title("Mel Spectrogram")
    plt.ylabel("Mel bins")
    plt.xlabel("Time")
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.show()


def audio_to_mel(audio: np.array, sr: int, n_mels: int) -> torch.tensor:
    """
    converts audio to a mel spectrogram

    audio  -- np.array of loaded audio values
    sr     -- int that determines the sample rate for mel spectrogram conversion
    n_mels -- number of mel spectrogram bins
    """
    trans = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=n_mels)
    mel_spectrogram = trans(torch.tensor(audio))    
    return mel_spectrogram


def plot_audio(audio: np.array) -> None:
    """
    plot the audio waveform
    """
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Normalized amplitude")
    plt.plot(audio[0][::10], color="teal")
    plt.show()


def convert_to_DB(
        mel_spectrogram: np.array, 
        multiplier: int, 
        amin: float, 
        db_multiplier: float, 
        top_db: float
        ) -> np.array:
    """
    converts amplitude values to decibles

    mel_spectrogram -- np.array of spectrogram values
    multiplier      -- 10 used for power, 20 for amplitude
    amin            -- lower threshold value
    db_multiplier   -- scaling factor
    top_db          -- minimum negative cut-off in db
    """
    mel_spectrogram_db = torchaudio.functional.amplitude_to_DB(
        x=mel_spectrogram, 
        multiplier=multiplier, 
        amin=amin, 
        db_multiplier=db_multiplier, 
        top_db=top_db
        )
    return mel_spectrogram_db


def norm_sg(spectrogram: np.array) -> np.array:
    """
    normalizes spectrogram values
    """
    s_db = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
    return s_db


def norm_midi_pitch(midi_pitches):
    """
    normalizes midi pitch values
    """
    return midi_pitches / 127


def save_array(arr: np.array, save_fp: str) -> None:
    """
    saves numpy array
    """
    np.save(file=save_fp, arr=arr)


def extract_chunk(
        piano_roll: np.array, 
        norm_db: np.array, 
        start: int, 
        stop: int, 
        song_len: int
        ) -> np.array:
    """
    for a given time duration, this function extracts an equal amount from the
    piano roll and spectrogram vector, taking into account the different
    frame rates of the two vectors
    """
    pr_frames = piano_roll.shape[1]
    sg_frames = norm_db.shape[1]

    # milliseconds per frame
    pr_rate = song_len / pr_frames 
    sg_rate = song_len / sg_frames

    start_pr = round(start * pr_rate)
    end_pr = round(stop * pr_rate)

    start_sg = round(start * sg_rate)
    end_sg = round(stop * sg_rate)

    pr_chunk = piano_roll[:, start_pr:end_pr]
    sg_chunk = norm_db[:, start_sg:end_sg].numpy() 

    return pr_chunk, sg_chunk


def create_tensor(
        debug_vis: bool, 
        audio_path: str, 
        midi_path: str, 
        sr: int,
        chunk_len: int,
        save_path: Path
        ) -> torch.tensor:
    """
    creates a tensor from an audio and midi file

    debug_vis  -- boolean that determines if visualizations 
                  are done for debugging purposes
    audio_path -- path to the .wav file
    midi_path  -- path to the .midi file
    """
    audio, _ = load_wav(audio_path, sample_rate=sr)
    song_len = len(audio)
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    piano_roll = midi.get_piano_roll(fs=100)

    if debug_vis:
        plot_audio(audio=audio)

    mel_spectrogram = audio_to_mel(audio=audio, sr=sr, n_mels=40)
    mel_spectrogram_db = convert_to_DB(
        mel_spectrogram=mel_spectrogram,
        multiplier=10, 
        amin=1e-10,
        db_multiplier=0.0,
        top_db=80.0
        )
    norm_db = norm_sg(mel_spectrogram_db)

    for m_sec in range(0, song_len, chunk_len):
        pr_chunk, sg_chunk = extract_chunk(
            piano_roll=piano_roll,
            norm_db=norm_db,
            start=m_sec,
            stop=m_sec+chunk_len,
            song_len=song_len
        )
        save_array(
            arr=sg_chunk,
            save_fp=str(save_path.joinpath(f"{m_sec}-{m_sec+chunk_len}-{audio_path.stem}-spec.npy"))
        )
        save_array(
            arr=pr_chunk,
            save_fp=str(save_path.joinpath(f"{m_sec}-{m_sec+chunk_len}-{audio_path.stem}-piano.npy"))
        )

    if debug_vis:
        plot_spectrogram(mel_spectrogram_db.squeeze(0))
        plt.show()


def convert_folder(audio_midi_dir: Path, sr: int, chunk_len: int, save_path: Path) -> None:
    """
    converts competition year folder of .wav and .midi files to tensors
    """
    audio_list = [x for x in audio_midi_dir.iterdir() if ".wav" in x.name]
    for audio_path in tqdm(audio_list, desc="Audio files processed"):
        matching_midi_path = [x for x in audio_midi_dir.glob(f"*{audio_path.stem}.midi")][0]
        create_tensor(
            debug_vis=False,
            audio_path=audio_path,
            midi_path=matching_midi_path,
            sr=sr,
            chunk_len=chunk_len, # in seconds
            save_path=save_path
            )


def main():
    args = parse_cla()
    for year_folder in args.maestro_dir.iterdir():
        if year_folder.is_dir():
            convert_folder(
                audio_midi_dir=args.maestro_dir.joinpath(year_folder), 
                sr=args.sr,
                chunk_len=args.chunk_len,
                save_path=args.ds_save_path
                )


if __name__ == "__main__":
    main()
