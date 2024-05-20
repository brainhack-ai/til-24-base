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
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = 'facebook/wav2vec2-large-robust-ft-swbd-300h'
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.data_dir = Path("data")

        data = {'key': [], 'audio': [], 'transcript': []}
        with jsonlines.open(self.data_dir / "asr.jsonl") as reader:
            for obj in reader:
                if (len(data['key']) > 10):
                    break
                for key, value in obj.items():
                    data[key].append(value)

        dataset = Dataset.from_dict(data)

        dataset = dataset.shuffle(seed=42)

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, train_size + val_size + test_size))

        train_dataset = train_dataset.map(self.preprocess_data, batched=True, batch_size=1, remove_columns=train_dataset.column_names)
        val_dataset = val_dataset.map(self.preprocess_data, batched=True, batch_size=1, remove_columns=val_dataset.column_names)
        test_dataset = test_dataset.map(self.preprocess_data, batched=True, batch_size=1, remove_columns=test_dataset.column_names)

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="steps",
            learning_rate=1e-4,
            per_device_train_batch_size=1,
            num_train_epochs=2,
            weight_decay=0.005,
            save_steps=500,
            eval_steps=500,
            logging_steps=10,
            load_best_model_at_end=True
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.processor.feature_extractor
        )

        self.trainer.train()



    def transcribe(self, audio_bytes: bytes) -> str:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        waveform = waveform.to(self.device)
        input_values = self.processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values.to(self.device)


        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription

    def preprocess_data(self, examples):
        input_values = []
        attention_masks = []
        labels = []

        for audio_path, transcript in zip(examples['audio'], examples['transcript']):
            speech_array, sampling_rate = torchaudio.load(self.data_dir / 'audio' / audio_path)
            speech_array = speech_array.squeeze(0)
            processed = self.processor(speech_array.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt", padding=True)

            with self.processor.as_target_processor():
                label = self.processor(transcript, return_tensors="pt", padding=True)

            input_values.append(processed.input_values.squeeze(0))
            attention_mask = torch.ones_like(processed.input_values)
            attention_mask[processed.input_values == self.processor.tokenizer.pad_token_id] = 0  
            attention_masks.append(attention_mask.squeeze(0))

            padded_label = torch.full(processed.input_values.shape[1:], -100, dtype=torch.long)
            actual_length = label.input_ids.shape[1]
            padded_label[:actual_length] = label.input_ids.squeeze(0)
            labels.append(padded_label)

        examples['input_values'] = torch.stack(input_values)
        examples['attention_mask'] = torch.stack(attention_masks)
        examples['labels'] = torch.stack(labels)

        return examples

def main():
    asr_manager = ASRManager()


    file_path = "data/audio/audio_2.wav"
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
    transcription = asr_manager.transcribe(audio_bytes)
    print("Transcription: ", transcription)



if __name__ == "__main__":
    main()
