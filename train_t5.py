from transformers import T5ForConditionalGeneration, T5Config
import torch
import numpy as np
import torch.nn as nn


def model_summary(model):
    """
    prints the size for each of the parameters in the model
    """
    for param in model.parameters():
        print(param.shape)


class audioT5(nn.Module):
    def __init__(self, feature_extractor, t5):
        super().__init__()
        self.fe = feature_extractor
        self.t5 = t5

    def forward(self, x):
        x = self.fe(x)
        x = self.t5(x)
        return x


def define_model():
    """
    defines simple feature extractor and transformer model
    """
    feature_extractor = nn.Linear(in_features=5000, out_features=512)
    t5_config = T5Config()
    t5 = T5ForConditionalGeneration(t5_config)
    model = audioT5(feature_extractor=feature_extractor, t5=t5)
    return model


def main():
    sg_arr = np.load("C:\\personal_ML\\music-transcription\\save\\0-5000-MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav-spec.npy")
    pr_arr = np.load("C:\\personal_ML\\music-transcription\\save\\0-5000-MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav-piano.npy")

    device = torch.device("cpu")

    sg_tensor = torch.tensor(sg_arr).to(device, dtype=torch.float32)
    pr_tensor = torch.tensor(pr_arr).to(device)

    model = define_model().to(device, dtype=torch.float32)

    print(model(sg_tensor))


if __name__ == "__main__":
    main()
