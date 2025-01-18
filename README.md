# NylonAMT
This project uses GPT2 in order to automatically transcribe classical guitar audio files into sheet music. This is done by first training a model to output a vector representing a piano roll given a mel spectrogram. The MAESTRO dataset found [here](https://magenta.tensorflow.org/datasets/maestro) is made from pairs of audio and MIDI data (aligned with ~3ms accuracy) from multiple years of piano recitals, totaling 200 hours. This was converted into input and output pair tensors with dataset_creation.py

[Alphatab](https://github.com/CoderLine/alphaTab) will be used for tab visualization.

![image](https://github.com/user-attachments/assets/f86ede41-5993-40af-a636-491630587b08)

![image](https://github.com/user-attachments/assets/c25051df-08de-4869-b4a2-2c5641f1ac3c)
