import jsonlines
import torchaudio
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from pathlib import Path
import torch
import librosa
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

class ASRManager:
    def __init__(self):
        # initialize the model here
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = 'facebook/wav2vec2-large-robust-ft-swbd-300h'
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)


    def transcribe(self, audio_bytes: bytes) -> str:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        waveform = waveform.to(self.device)
        input_values = self.processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values.to(self.device)


        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription


def main():
    asr_manager = ASRManager()


    file_path = "data/audio/audio_3499.wav"
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
    transcription = asr_manager.transcribe(audio_bytes)
    print("Transcription: ", transcription)



if __name__ == "__main__":
    main()
