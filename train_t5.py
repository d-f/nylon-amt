from mido import MidiFile
import matplotlib.pyplot as plt
# from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import torch
from scipy.io import wavfile
import numpy as np
import torchaudio



def resample_audio(audio, target_rate, og_rate):
    trans = torchaudio.transforms.Resample(og_rate, target_rate)
    audio = trans(audio)
    return audio

def convert_to_mono(stereo_audio):
    return torch.mean(stereo_audio, dim=0, keepdim=True)


def load_wav(file_path):
    sr, audio = torchaudio.load(file_path, normalize=True)
    return audio, sr


def plot_spectrogram(specgram):
    plt.title("Mel Spectrogram")
    plt.ylabel("Hz")
    plt.xlabel("Time")
    plt.imshow(specgram, origin="lower", aspect="auto")



def audio_to_mel(audio, sr):
    trans = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=512)
    mel_spectrogram = trans(torch.tensor(audio))
    
    return mel_spectrogram


def prepare_input(mel_spectrogram, tokenizer):

    input_str = ' '.join(map(str, mel_spectrogram.flatten().tolist()))
    

    inputs = tokenizer(input_str, return_tensors="pt", padding=True, truncation=True)
    return inputs


def plot_audio(audio):
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Normalized amplitude")
    plt.plot(audio[0][::10], color="teal")
    plt.show()


def convert_to_DB(mel_spectrogram):
    mel_spectrogram_db = torchaudio.functional.amplitude_to_DB(x=mel_spectrogram, multiplier=10, amin=1e-10, db_multiplier=0.0, top_db=80.0)
    return mel_spectrogram_db


def main():
    debug_vis = True
    sr, audio = load_wav("C:\\personal_ML\\music-transcription\\maestro-v3.0.0\\maestro-v3.0.0\\2004\\MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav")
    

    # mid = MidiFile('C:\\personal_ML\\music-transcription\\maestro-v3.0.0\\maestro-v3.0.0\\2004\\MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi', clip=True)
    # print(mid)
    
    audio = convert_to_mono(stereo_audio=audio)

    if sr != 16000:
        audio = resample_audio(audio=audio, target_rate=16000, og_rate=sr)

    if debug_vis:
        plot_audio(audio=audio)

    # Convert audio to mel spectrogram
    mel_spectrogram = audio_to_mel(audio, 16000)
    mel_spectrogram_db = convert_to_DB(mel_spectrogram)
    if debug_vis:
        plot_spectrogram(mel_spectrogram_db.squeeze(0))
        plt.show()












    # Prepare input for the T5 model
    inputs = prepare_input(mel_spectrogram)
    
    # Generate transcription
    output_sequences = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=512)
    
    # Decode the output
    transcription = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    config = T5Config()


    model = T5ForConditionalGeneration(config=config)
    
    return transcription


if __name__ == "__main__":
    main()
