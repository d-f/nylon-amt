from mido import MidiFile
import matplotlib.pyplot as plt
# from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import torch
from scipy.io import wavfile
import numpy as np
import torchaudio
import librosa
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist


def load_wav(file_path):
    audio, sr = librosa.load(file_path, mono=True, sr=16000)
    return audio, sr


def plot_spectrogram(spectrogram):
#     mel_bins = spectrogram.mel_scale.fb.size(0)
#     mel_freqs = torchaudio.functional.melscale_frequencies(
#     n_mels=mel_bins,
#     f_min=0.0,
#     f_max=16000 / 2.0,
# )
    plt.title("Mel Spectrogram")
    plt.ylabel("Mel bins")
    plt.xlabel("Time")
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    # plt.figure(figsize=(10, 4))
    # plt.imshow(spectrogram, origin='lower', aspect='auto', cmap='viridis')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel spectrogram (Hz scale)')
    # plt.xlabel('Time')
    # plt.ylabel('Frequency (Hz)')
    # plt.yticks(np.linspace(0, mel_bins - 1, num=10), np.round(mel_freqs[np.linspace(0, mel_bins - 1, num=10).astype(int)], 1))
    # plt.tight_layout()
    plt.show()


def audio_to_mel(audio, sr):
    trans = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=40)
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


def norm_sg(spectrogram):
    s_db = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
    return s_db


def norm_midi_pitch(midi_pitches):
    return midi_pitches / 127


def align(audio, sr, midi):
    midi_onsets = []
    midi_pitches = []
    current_time = 0

    for msg in midi:
        current_time += msg.time
        if not msg.is_meta and msg.type == "note_on" and msg.velocity > 0:
            midi_onsets.append(current_time)
            midi_pitches.append(msg.note)

    midi_onsets = np.array(midi_onsets)
    midi_pitches = np.array(midi_pitches)

    # Audio Onset Detection
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, max_size=1, lag=1)

    # Normalize the onset envelope
    onset_env = librosa.util.normalize(onset_env)

    # Apply a median filter to smooth the onset envelope
    onset_env = median_filter(onset_env, size=3)

    # Adjust the parameters for peak picking to capture more onsets initially
    pre_max = 3
    post_max = 3
    pre_avg = 3
    post_avg = 3
    delta = 0.01  # Lower threshold to capture more peaks
    wait = 1

    # Use peak picking directly on the onset envelope
    peaks = librosa.util.peak_pick(onset_env, pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg, delta=delta, wait=wait)
    audio_onsets_initial = librosa.frames_to_time(peaks, sr=sr)

    # Use DTW to align MIDI onsets to audio onsets
    distance, path = dtw(midi_onsets, audio_onsets_initial, dist=cdist)

    # Extract the aligned audio onsets
    aligned_audio_onsets = [audio_onsets_initial[j] for (i, j) in path]
    # midi_onsets = []
    # midi_pitches = []

    # current_time = 0

    # for msg in midi:
    #     current_time += msg.time
    #     if not msg.is_meta and msg.type == "note_on" and msg.velocity > 0:
    #         midi_onsets.append(current_time)
    #         midi_pitches.append(msg.note)

    # midi_onsets = np.array(midi_onsets)
    # midi_pitches = np.array(midi_pitches)




    # onset_env = librosa.onset.onset_strength(y=audio, sr=sr, max_size=1, lag=1)

    # # Normalize and smooth the onset envelope
    # onset_env = librosa.util.normalize(onset_env)
    # # onset_env = librosa.effects.harmonic(onset_env)

    # # Adjust the threshold and sensitivity
    # # onset_threshold = np.mean(onset_env) + np.std(onset_env)  # Lowered threshold
    # # audio_onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time', backtrack=True, delta=onset_threshold, wait=1)

    # # Use peak picking directly on the onset envelope
    # peaks = librosa.util.peak_pick(onset_env, pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.0005, wait=1)
    # audio_onsets = librosa.frames_to_time(peaks, sr=sr)

    # pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
    # onset_frames = librosa.time_to_frames(audio_onsets, sr=sr)
    # # audio_pitches = []

    # onset_frames = onset_frames[onset_frames < pitches.shape[1]]

    # extracted_pitches = []
    # for onset in onset_frames:
    #     # frame = librosa.time_to_frames(onset, sr=sr)
    #     # print(frame)
    #     pitch_values = pitches[:, onset]
    #     magnitude_values = magnitudes[:, onset]
    #     max_magnitude_index = np.argmax(magnitude_values)
    #     extracted_pitches.append(pitch_values[max_magnitude_index])

    # for t in range(pitches.shape[1]):
    #     index = magnitudes[:, t].argmax()
    #     pitch = pitches[index, t]
    #     if pitch > 0:
    #         audio_pitches.append(pitch)
    #     else:
    #         audio_pitches.append(0)


    from scipy.spatial.distance import cdist
    from librosa.sequence import dtw

    midi_features = np.column_stack((midi_onsets, midi_pitches))
    audio_features = np.column_stack((audio_onsets, extracted_pitches))

    distance_matrix = cdist(midi_features, audio_features, metric='euclidean')
    D, wp = dtw(C=distance_matrix, subseq=True)

    # Extract aligned sequences
    aligned_midi_onsets = midi_onsets[wp[:, 0]]
    aligned_midi_pitches = midi_pitches[wp[:, 0]]
    aligned_audio_onsets = audio_onsets[wp[:, 1]]
    aligned_audio_pitches = extracted_pitches[wp[:, 1]]


    # norm_spectrogram = norm_sg(spectrogram)

    # midi_pitches = norm_midi_pitch(midi_pitches)

    # norm_spectrogram = norm_spectrogram.flatten()

    # midi_sequence = ["[MID]"] + [str(pitch) for pitch in midi_pitches]

    # audio_sequence = ["[AUD]"] + [str(bin_value) for bin_value in norm_spectrogram]
    
    pass

def main():
    debug_vis = False
    audio, sr = load_wav("C:\\personal_ML\\music-transcription\\maestro-v3.0.0\\maestro-v3.0.0\\2004\\MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav")
    

    midi = MidiFile('C:\\personal_ML\\music-transcription\\maestro-v3.0.0\\maestro-v3.0.0\\2004\\MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi', clip=True)
    # print(mid)
    
    # audio = convert_to_mono(stereo_audio=audio)

    
    
    # if sr != 16000:
    #     audio = resample_audio(audio=audio, target_rate=16000, og_rate=sr)

    if debug_vis:
        plot_audio(audio=audio)

    audio, midi = align(audio=audio, midi=midi, sr=sr)

    # Convert audio to mel spectrogram
    mel_spectrogram = audio_to_mel(audio, 16000)
    mel_spectrogram_db = convert_to_DB(mel_spectrogram)
    if debug_vis:
        plot_spectrogram(mel_spectrogram_db.squeeze(0))
        plt.show()

    



if __name__ == "__main__":
    main()
