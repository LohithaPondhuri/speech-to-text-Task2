# speech-to-text-Task2

# README.md

## 🎤 Speech-to-Text System (Using SpeechRecognition & Wav2Vec2)

This project demonstrates a basic **Speech-to-Text (STT)** system using two popular approaches:
- **Online API-based**: Using the `SpeechRecognition` library with Google Web Speech API.
- **Offline Deep Learning-based**: Using **Facebook's Wav2Vec2** model via Hugging Face Transformers.

### ✅ Deliverable
> A functional Python system capable of transcribing short audio clips into text using pre-trained models.

## ⚙️ Requirements

```bash
pip install SpeechRecognition pyaudio transformers torchaudio soundfile
---

## ▶️ Method 1: Using `SpeechRecognition` (Online, Live Mic Input)

```bash
python recognize_with_speechrecognition.py
```
Then speak into your microphone when prompted.

## ▶️ Method 2: Using `Wav2Vec2` (Offline, File Input)

```bash
python transcribe_with_wav2vec2.py
```
Make sure you have a valid WAV file as `sample_audio.wav` (mono, 16kHz).

## 🧪 Sample Audio Preparation

You can record or convert audio using FFmpeg:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 sample_audio.wav

## 📜 License

MIT License
