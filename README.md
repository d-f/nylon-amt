# NylonHFTT

This project makes use of hierarchical frequency time transformers in order to transcribe classical guitar audio into sheet music and guitar tabulature. The original implementation of this paper can be found at [https://github.com/sony/hFT-Transformer/](https://github.com/sony/hFT-Transformer/) and [https://arxiv.org/abs/2307.04305](https://arxiv.org/abs/2307.04305)

First, a model is trained to transcribe solo piano recital pieces using the MAESTRO dataset
which can be found here: [https://magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro) This dataset includes paired audio and midi files that are aligned within 3ms. There are data for 10 years of international competitive piano recitals created by recording MIDI alongside audio at the time of the performance.

[Alphatab](https://github.com/CoderLine/alphaTab) will be used for tab visualization.
