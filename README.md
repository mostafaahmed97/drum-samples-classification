# Drum Sample Classification

A simple project that classifies audio samples as any of the following classes: kick, snare or hihat.

Audio samples are pre processed through the following steps:
- Truncating samples to fixed length
- Extracting the following features from each sample
  - Zero crossing
  - Spectral centroid
  - Spectral rollof
  - MFCC
 
Models evaluated for task are MLP, SVM & Random Forest. SVM with polynomial kernel showed best performance for task.
