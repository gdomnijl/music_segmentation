# music_segmentation
282 final project
link to SALAMI data (https://github.com/DDMAL/salami-data-public)

**Project timeline: May 5th Project due**

## Todos:
1. Data collection **By 19th Apr (Week 1 Fri)**

  1.1 Set up cloud instance **Nami**
  1.2 Download audio files from SALAMI **Jon**

  1.3 Download labels and mark boundary on 4-second window **Jinlin**

  1.4 Transform audio files to spectrogram, format it to vector form each frame **Jinlin**

1.5 Feed through an identity map to check

2. Building network architecture **By 21st Apr (Week 1 Sun)**
2.1 LSTM
2.2 Attention (encoder-only self-attention?)

3. Training **By 26th Apr (Week 2 Fri)**

4. Evaluation **By 4rd May (Week 3 Fri)**
4.1 Distance metric (BLEU-like precision metric: overlap with ground truth/prediction sequence length) for first model (boundary or not-boundary)
4.2 Cross entropy loss for second model (label/prediction discrepency)

### Notes on network structure
Two different models: one model to predict where the boundary of section is (a pair of timestamp: start and end) to segment the music into sections; the subsequent model to classify each section as one of the section label.

We need to train these two models sequentially because we need a good enough boundary-predicting model in order to go about section classification. 

1. First model (boundary model)
*input: spectrogram at each time step
*output: (start, end) - two timestamp of where boundary starts and ends
*label: (start, end) - computed from 4-second-window centered at the section onset time (e.g. data looks like 1:11 Bridge, 2:40 Chorus; boundary is (1:09, 1:13), (2:38, 2:42))

2. Second model (section model) - sequential version 
*input: spectrogram at each time step, if frames fall into previously predicted boundary range, replace it with a delimiter/token vector representation (ASK**)
*intermediate hidden state: representation of the section is captured using the last time step activation before any boundary delimiter; 
*output: each section representation generates a prediction of what that section is.
*label: section label

## Questions:
What is a good size for frame/time step of spectrogram? 

