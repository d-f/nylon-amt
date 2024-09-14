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
    parser.add_argument("-midi_dir", type=Path)
    parser.add_argument("-audio_dir", type=Path)
    parser.add_argument("-sr", type=int)
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


def create_tensor(debug_vis: bool, audio_path: str, midi_path: str, sr: int) -> torch.tensor:
    """
    creates a tensor from an audio and midi file

    debug_vis  -- boolean that determines if visualizations 
                  are done for debugging purposes
    audio_path -- path to the .wav file
    midi_path  -- path to the .midi file
    """
    audio = load_wav(audio_path, sr=sr)

    midi = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = midi.get_piano_roll(fs=int(1/0.01))

    if debug_vis:
        plot_audio(audio=audio)

    mel_spectrogram = audio_to_mel(audio=audio, sr=16000, n_mels=40)
    mel_spectrogram_db = convert_to_DB(
        mel_spectrogram=mel_spectrogram,
        multiplier=10, 
        amin=1e-10,
        db_multiplier=0.0,
        top_db=80.0
        )
    norm_db = norm_sg(mel_spectrogram_db)
    
    if debug_vis:
        plot_spectrogram(mel_spectrogram_db.squeeze(0))
        plt.show()


def convert_dataset(audio_dir: Path, midi_dir: Path, sr: int) -> None:
    """
    converts dataset of .wav and .midi files to tensors
    """
    for audio_path in audio_dir.iterdir():
        matching_midi_path = midi_dir.glob(f"*{audio_path.name}.midi")[0]
        create_tensor(
            debug_vis=False,
            audio_path=audio_path,
            midi_path=matching_midi_path,
            sr=sr
            )


def main():
    args = parse_cla()
    convert_dataset(audio_dir=args.audio_dir, midi_dir=args.midi_dir, sr=args.sr)


if __name__ == "__main__":
    main()
