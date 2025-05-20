On-Device Auditory System for the BEATRIX Humanoid Head

## Overview

This repository contains the code, and hardware designs for adding a fully on-device auditory perception and conversational intelligence system to the open-source BEATRIX humanoid head. The system enables 360° sound source localization (SSL), speech recognition, and dynamic dialogue—all running locally on a Raspberry Pi-powered platform without cloud services.

Key components:

* **Preprocessing & Feature Extraction**:

  * `GCCPHAT_ZeroPad_Freqrange.py`: GCC-PHAT cross-correlation with zero-padding and PHAT weighting to estimate interaural time differences (ITD).
  * `Gammatone_spectrogram_Function.py`: Gammatone filterbank spectrogram computation emulating cochlear frequency resolution.
  * `Wav_splitter.py`: Automatic splitting and band-pass filtering of long binaural recordings into 0.5 s segments with voice activity detection.
* **Neural Network Training**:

  * `Main_Training_Script.py`: End-to-end PyTorch training pipeline combining ITD (GCC-PHAT) and interaural level & spectral cues (gammatone spectrograms). Supports CNN and GRU architectures, data augmentation with noise, normalisation, learning-rate scheduling, early stopping, and evaluation on held-out and completely unseen datasets.
* **Neural Network Running**:

  * `Live_DNN_runner_Pi.py`: Continuously records audio on the Pi, applies band-pass and resampling, detects activity, extracts GCC-PHAT & Gammatone features, runs the trained DNN for real-time SSL, visualizes results, and commands Arduino stepper motors to track the source.
* **Conversational LLM**:

* `LLM_Server.py`: Lightweight Flask API that forwards incoming text via HTTP POST to a local Ollama LLM (“BEATRIX” model) and returns its generated response as JSON.
* `Live_LLM_runner_Pi.py`: Continuously records stereo audio until end-of-speech silence using a Butterworth band-pass filter, selects the loudest channel for on-device Whisper transcription, sends the text via HTTP POST to a remote Flask LLM server, and vocalizes replies locally with Piper TTS.

## Requirements

* **Hardware**: Raspberry Pi 5, 2× SPH0645LM4H I²S microphones, 3D-printed ears and mounts, BEATRIX head.
* **Software**: Python 3.8+, PyTorch, NumPy, SciPy, Librosa, SoundFile, Faster-Whisper, Ollama, Piper TTS.

Enable I²S mic in `/boot/firmware/config.txt`:

```ini
dtoverlay=googlevoicehat-soundcard
```

## Results & Evaluation

* **SSL Performance**: Mean absolute error (MAE) of 5.6° on seen configurations; 20.3° on novel head-speaker positions; 34.4° in unseen rooms.
* **ASR Accuracy**: \~94% transcription accuracy with small.en Faster-Whisper.
* **LLM Response**: Sub-0.05 s per word TTS via Piper.

Detailed plots and evaluation metrics are in the `examples/` folder.

## Citation

If you use this work, please cite:

> Shadar, J. (2025). BEATRIX - Artificial Auditory System . MEng Final Project, University of Bath.
