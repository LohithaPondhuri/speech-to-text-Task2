import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import os

def transcribe(file_path):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    waveform, sample_rate = torchaudio.load(file_path)

    # Resample if not 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Tokenize and predict
    input_values = tokenizer(waveform.squeeze().numpy(), return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])

    return transcription

if __name__ == "__main__":
    # Change this to the path of your local .wav file
    file_path = "sample_audio.wav"
    
    if os.path.exists(file_path):
        print("📝 Transcription:", transcribe(file_path))
    else:
        print(f"❌ File not found: {file_path}")
