 import argparse
import os
import torch
from torch.utils.data import Dataset
from transformers import ErnieTokenizer, ErnieForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

class TextClassificationPrediction:
    def __init__(self, model_dir, vocab_path, inference_device, runtime_backend, batch_size, sequence_length, logging_interval, use_fp16, use_fast_tokenizer):
        self.tokenizer = ErnieTokenizer.from_pretrained(vocab_path) if use_fast_tokenizer else AutoTokenizer.from_pretrained(model_dir)
        self.model = ErnieForSequenceClassification.from_pretrained(model_dir) if use_fast_tokenizer else AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(inference_device)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.logging_interval = logging_interval
        self.use_fp16 = use_fp16
        self.device = inference_device

    def preprocess(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.sequence_length, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def postprocess(self, input_ids, predictions):
        logits = predictions.logits
        predicted_labels = torch.argmax(logits, dim=-1)
        confidences = torch.softmax(logits, dim=-1)[:, 1]
        output = [{"batch_id": batch_id, "example_id": example_id, "input_sentence": input_sentence, "predicted_label": label.item(), "confidence_score": confidence.item()} for batch_id, example_ids in enumerate(input_ids) for input_sentence, label, confidence in zip(example_ids["input_ids"].tolist(), predicted_labels[batch_id], confidences[batch_id])]
        return output

    def predict(self, texts):
        inputs = self.preprocess(texts)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            if self.use_fp16:
                self.model, inputs = self.model.half(), {k: v.half() for k, v in inputs.items()}
            predictions = self.model(**inputs)
        return self.postprocess(input_ids, predictions)

def batchify(dataset, batch_size):
    data_len = len(dataset)
    batches = [dataset[i * batch_size: (i + 1) * batch_size] for i in range(data_len // batch_size)]
    if data_len % batch_size != 0:
        batches.append(dataset[data_len - data_len % batch_size:])
    return batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Model directory")
    parser.add_argument("--vocab_path", help="Tokenizer vocab path")
    parser.add_argument("--inference_device", default="cuda", help="Inference device")
    parser.add_argument("--runtime_backend", help="Runtime backend")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--logging_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 mode")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Use fast tokenizer")
    args = parser.parse_args()

    prediction_class = TextClassificationPrediction(args.model_dir, args.vocab_path, args.inference_device, args.runtime_backend, args.batch_size, args.sequence_length, args.logging_interval, args.use_fp16, args.use_fast_tokenizer)

    # Assume we have a list of texts to predict
    texts = [
        "Text to classify 1",
        "Text to classify 2",
        # ...
    ]

    batches = batchify(texts, args.batch_size)
    for batch_id, batch in enumerate(tqdm(batches)):
        predictions = prediction_class.predict(batch)
        for prediction in predictions:
            print(prediction)

if __name__ == "__main__":
    main()