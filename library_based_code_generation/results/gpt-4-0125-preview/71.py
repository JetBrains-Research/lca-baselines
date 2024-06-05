```python
import argparse
import torch
from transformers import ErnieTokenizer

def batchify(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

class ErnieSequenceClassifier:
    def __init__(self, model_dir, tokenizer_vocab_path, device, backend, use_fp16, use_fast_tokenizer):
        self.device = device
        self.use_fp16 = use_fp16
        self.tokenizer = ErnieTokenizer.from_pretrained(tokenizer_vocab_path, use_fast=use_fast_tokenizer)
        self.model = torch.jit.load(model_dir, map_location=device)
        if backend == "cuda":
            self.model.to(device)
        if use_fp16:
            self.model.half()
    
    def preprocess(self, texts, max_length):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    def inference(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs
    
    def postprocess(self, inference_output):
        logits = inference_output.logits
        probabilities = torch.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probabilities, dim=-1)
        return predictions.cpu().numpy(), confidences.cpu().numpy()
    
    def predict(self, texts, max_length):
        inputs = self.preprocess(texts, max_length)
        outputs = self.inference(inputs)
        predictions, confidences = self.postprocess(outputs)
        return predictions, confidences

def main():
    parser = argparse.ArgumentParser(description="Sequence Classification Prediction")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the model")
    parser.add_argument("--tokenizer_vocab_path", type=str, required=True, help="Path to tokenizer vocabulary")
    parser.add_argument("--inference_device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--runtime_backend", type=str, default="cpu", choices=["cpu", "cuda"], help="Runtime backend")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--sequence_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--logging_interval", type=int, default=10, help="Interval for logging")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 mode for inference")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Use fast tokenizer")
    args = parser.parse_args()

    classifier = ErnieSequenceClassifier(args.model_dir, args.tokenizer_vocab_path, args.inference_device, args.runtime_backend, args.use_fp16, args.use_fast_tokenizer)
    
    texts = ["This is a test.", "Another test sentence."]
    batches = batchify(texts, args.batch_size)
    
    for batch_id, batch in enumerate(batches):
        predictions, confidences = classifier.predict(batch, args.sequence_length)
        for example_id, (text, prediction, confidence) in enumerate(zip(batch, predictions, confidences)):
            print(f"Batch {batch_id}, Example {example_id}, Input: '{text}', Predicted Label: {prediction}, Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
```