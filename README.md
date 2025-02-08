# NylonHFTT

This project makes use of hierarchical frequency time transformers in order to transcribe classical guitar audio into sheet music and guitar tabulature. The original implementation of this model can be found at [https://github.com/sony/hFT-Transformer/](https://github.com/sony/hFT-Transformer/) and [https://arxiv.org/abs/2307.04305](https://arxiv.org/abs/2307.04305)

First, a model is trained to transcribe solo piano recital pieces using the MAESTRO dataset
which can be found here: [https://magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro) This dataset includes paired audio and midi files that are aligned within 3ms. There are data for 10 years of international competitive piano recitals created by recording MIDI alongside audio at the time of the performance.

dataset_creation.py was used in order to format the MAESTRO dataset in the expected way. The HFTT github was used with very minor adjustments

Changes to the HFTT repo:
- m_training.py changing device to torch.device("cuda") compared to just "cuda" on line 111
- train.py changing the total to be the length of the iterator rather than a fixed value in the tqdm progress bar on line 16.

m_training.py was used in order to train the model, with example usage in the file /evaluation/EXE-EVALUATION-MAESTRO.sh

[Alphatab](https://github.com/CoderLine/alphaTab) will be used for tab visualization.
