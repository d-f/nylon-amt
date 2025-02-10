This project makes use of hierarchical frequency time transformers in order to transcribe classical guitar audio into sheet music and guitar tabulature. The original implementation of this model can be found at [https://github.com/sony/hFT-Transformer/](https://github.com/sony/hFT-Transformer/) and [https://arxiv.org/abs/2307.04305](https://arxiv.org/abs/2307.04305)

First, a model is trained to transcribe solo piano recital pieces using the MAESTRO dataset
which can be found here: [https://magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro) This dataset includes paired audio and midi files that are aligned within 3ms. There are data for 10 years of international competitive piano recitals created by recording MIDI alongside audio at the time of the performance.

# Training details
The following model architecture hyperparameters were reduced, otherwise the architecture was identical to the one presented in the paper:
| hidden layer | position-wise feed forward layer | num encoder layers | num decoder layers | num encoder heads | num decoder heads | 
| ------------ | -------------------------------- | ------------------ | ------------------ | ----------------- | ----------------- | 
| 64           | 128                              | 2                  | 2                  | 2                 | 2                 |

| batch size | learning rate | drop out | dataset size     | num epochs | 
| ---------- | ------------- | -------- | ---------------- | ---------- |
| 8          | 1e-4          | 0.1      | 0.2% of original | 20         |

# MAESTRO dataset performance
| Precision | Recall | F1    | 
| --------- | ------ | ----- |
| 0.01      | 0.95   |  0.02 |

It is clear that more than 0.2% of the dataset needs to be used for training. 

dataset_creation.py was used in order to format the MAESTRO dataset in the expected way. The HFTT github was used with very minor adjustments

m_training.py was used in order to train the model, with example usage in the file /evaluation/EXE-EVALUATION-MAESTRO.sh

Changes to the HFTT repo:
- m_training.py
  - changing device to torch.device("cuda") compared to just "cuda" on line 111
  - moving testing to end of training
- train.py
  - changing the total to be the length of the iterator rather than a fixed value in the tqdm progress bar on line 16.
  -  measure precision, recall and f1 on test set with mir_eval

[Alphatab](https://github.com/CoderLine/alphaTab) will be used for tab visualization.
