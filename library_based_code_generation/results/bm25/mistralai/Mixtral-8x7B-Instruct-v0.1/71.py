 import argparse
import os
import torch
from transformers import ErnieTokenizer
from torchvision.transforms import batch_transforms

def batchify(data, batch_size, max_seq_length):
batch = []
texts = []
for text in data:
if len(texts) == batch_size or len(text) + len(texts[-1]) > max_seq_length:
batch_transforms = batch_transforms(
transforms.Pad(max_seq_length),
transforms.Collate( dental_collate),
)
batch = batch_transforms(batch)
yield batch, texts
batch = []
texts = [text]
else:
batch.append(text)

def predict(texts, model_dir, vocab_path, device, runtime_backend, batch_size, sequence_length, logging_interval, fp16, fast_tokenizer):
tokenizer = ErnieTokenizer.from_pretrained(vocab_path)
runtime = load_external_data_for_model(model_dir, device, runtime_backend, logging_interval, fp16)
input_ids, attention_masks = tokenizer(texts, padding=True, truncation=True, max_length=sequence_length, return_tensors="pt")
input_ids = input_ids.to(device)
attention_masks = attention_masks.to(device)
with torch.no_grad():
output = batch_predict(runtime, input_ids, attention_masks)
predictions = output.logits.argmax(dim=-1)
return predictions.cpu().numpy()

def main():
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--vocab_path", required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--runtime_backend", default="caffe2")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--sequence_length", type=int, default=128)
parser.add_argument("--logging_interval", type=int, default=100)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--fast_tokenizer", action="store_true")
args = parser.parse_args()

texts = ["This is a sample text.", "Another text to predict."]
predictions = predict(texts, args.model_dir, args.vocab_path, args.device, args.runtime_backend, args.batch_size, args.sequence_length, args.logging_interval, args.fp16, args.fast_tokenizer)

for i, (batch, texts) in enumerate(batchify(texts, args.batch_size, args.sequence_length)):
prediction = predictions[batch]
for j, text in enumerate(texts):
print(f"Batch ID: {i}, Example ID: {j}, Input: {text}, Predicted Label: {prediction[j]}, Confidence Score: {prediction[j]:.2f}")

if __name__ == "__main__":
main()