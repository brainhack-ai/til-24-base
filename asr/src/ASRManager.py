import jsonlines
import torchaudio
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from pathlib import Path
import torch
import librosa

class ASRManager:
    def __init__(self):
        # initialize the model here
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = 'facebook/wav2vec2-large-960h'
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.data_dir = Path("data")

        data = {'key': [], 'audio': [], 'transcript': []}
        with jsonlines.open(self.data_dir / "asr.jsonl") as reader:
            for obj in reader:
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
            num_train_epochs=3,
            weight_decay=0.005,
            save_steps=500,
            eval_steps=500,
            logging_steps=10,
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.processor.feature_extractor
        )

        trainer.train()



    def transcribe(self, audio_bytes: bytes) -> str:
        audio_input, sample_rate = librosa.load(audio_bytes, sr=16000)
        input_values = self.processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values


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