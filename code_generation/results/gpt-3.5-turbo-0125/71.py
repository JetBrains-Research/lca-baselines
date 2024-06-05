import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def batchify_text_data(texts, batch_size):
    # Function to batchify text data
    batched_texts = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    return batched_texts

class ErnieSequenceClassification:
    def __init__(self, model_dir, tokenizer_path, device, backend, batch_size, sequence_length, logging_interval, fp16_mode, fast_tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.logging_interval = logging_interval
        self.fp16_mode = fp16_mode
        self.fast_tokenizer = fast_tokenizer

    def preprocess_texts(self, texts):
        # Preprocess input texts
        encoded_texts = self.tokenizer(texts, padding=True, truncation=True, max_length=self.sequence_length, return_tensors='pt')
        return encoded_texts

    def inference(self, encoded_texts):
        # Perform inference
        with torch.inference_mode():
            outputs = self.model(**encoded_texts)
        return outputs

    def postprocess_inference(self, outputs):
        # Postprocess inference data
        predicted_labels = torch.argmax(outputs.logits, dim=1)
        confidence_scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        return predicted_labels, confidence_scores

    def predict(self, texts):
        encoded_texts = self.preprocess_texts(texts)
        outputs = self.inference(encoded_texts)
        predicted_labels, confidence_scores = self.postprocess_inference(outputs)
        return predicted_labels, confidence_scores

def main():
    parser = argparse.ArgumentParser(description='Sequence Classification Prediction using Ernie Model')
    parser.add_argument('--model_dir', type=str, help='Model directory path')
    parser.add_argument('--tokenizer_vocab_path', type=str, help='Tokenizer vocab path')
    parser.add_argument('--inference_device', type=str, help='Inference device (cpu or cuda)')
    parser.add_argument('--runtime_backend', type=str, help='Runtime backend')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--sequence_length', type=int, help='Sequence length')
    parser.add_argument('--logging_interval', type=int, help='Logging interval')
    parser.add_argument('--fp16_mode', action='store_true', help='Enable FP16 mode')
    parser.add_argument('--fast_tokenizer', action='store_true', help='Use fast tokenizer')
    args = parser.parse_args()

    texts = ["Example text 1", "Example text 2", "Example text 3"]  # Example input texts
    batched_texts = batchify_text_data(texts, args.batch_size)

    prediction_model = ErnieSequenceClassification(args.model_dir, args.tokenizer_vocab_path, args.inference_device, args.runtime_backend, args.batch_size, args.sequence_length, args.logging_interval, args.fp16_mode, args.fast_tokenizer)

    for batch_id, batch_texts in enumerate(batched_texts):
        predicted_labels, confidence_scores = prediction_model.predict(batch_texts)
        for example_id, (label, score) in enumerate(zip(predicted_labels, confidence_scores)):
            print(f"Batch ID: {batch_id}, Example ID: {example_id}, Input Sentence: {batch_texts[example_id]}, Predicted Label: {label}, Confidence Score: {score}")

if __name__ == "__main__":
    main()