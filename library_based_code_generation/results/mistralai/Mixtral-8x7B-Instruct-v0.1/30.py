from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

from transformers import (
BertConfig,
BertModel,
BertTokenizer,
BertTokenizerFast,
get_linear_schedule_with_warmup,
)
from transformers.file_utils import ModelOutput


@dataclass
class BatchEncoding:
input_ids: torch.Tensor
attention_mask: torch.Tensor
labels: torch.Tensor


def create_transformer_tagger_model(num_tags: int) -> torch.nn.Module:
config = BertConfig()
config.num_labels = num_tags
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
return model


def create_transformer_tokenizer(max_seq_length: int) -> BertTokenizerFast:
return BertTokenizerFast.from_pretrained("bert-base-uncased", max_length=max_seq_length)


def create_transformer_model(num_tags: int) -> torch.nn.Module:
config = BertConfig()
config.num_labels = num_tags
model = BertModel.from_pretrained("bert-base-uncased", config=config)
return model


def convert_transformer_inputs(tokenizer, sequence: str, max_seq_length: int) -> BatchEncoding:
encoding = tokenizer.batch_encode_plus(
[sequence],
max_length=max_seq_length,
pad_to_max_length=True,
return_attention_mask=True,
return_tensors="pt",
)
encoding["labels"] = torch.tensor([tokenizer.convert_tokens_to_ids(["[CLS]"] + sequence.split() + ["[SEP]"])])
return BatchEncoding(**encoding)


def convert_transformer_outputs(outputs: ModelOutput, num_tags: int) -> Tuple[torch.Tensor, ...]:
logits = outputs.logits
probs = torch.nn.functional.softmax(logits, dim=-1)
predictions = torch.argmax(probs, dim=-1)
return logits, probs, predictions


def evaluate_sequence(model, tokenizer, sequence: str, num_tags: int, max_seq_length: int) -> Tuple[torch.Tensor, ...]:
inputs = convert_transformer_inputs(tokenizer, sequence, max_seq_length)
outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask)
logits, probs, predictions = convert_transformer_outputs(outputs, num_tags)
return logits, probs, predictions


def group_pairs_into_minibatches(pair_list: List[Tuple[str, str]], batch_size: int, max_seq_length: int) -> List[List[Tuple[str, str]]]:
minibatches = []
current_minibatch = []
current_length = 0
for pair in pair_list:
current_minibatch.append(pair)
current_length += len(pair[0]) + len(pair[1])
if current_length > batch_size * max_seq_length:
minibatches.append(current_minibatch)
current_minibatch = []
current_length = 0
if current_minibatch:
minibatches.append(current_minibatch)
return minibatches


def main():
import argparse
import os
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
args = parser.parse_args()

config = torch.load(args.config)

model_config = config["model"]
optimizer_config = config["optimizer"]
learning_rate = config["learning_rate"]
training_params = config["training"]

num_tags = model_config["num_tags"]
max_seq_length = model_config["max_seq_length"]
batch_size = training_params["batch_size"]
num_epochs = training_params["num_epochs"]

model = create_transformer_tagger_model(num_tags).to(device)
tokenizer = create_transformer_tokenizer(max_seq_length)
optimizer = get_optimizer(optimizer_config["name"], model, **optimizer_config["params"])
scheduler = get_linear_schedule_with_warmup(
optimizer,
num_warmup_steps=int(len(train_data) * num_epochs * 0.1),
num_training_steps=int(len(train_data) * num_epochs),
)

train_data = load_dataset("train")
val_data = load_dataset("val")

for epoch in range(num_epochs):
model.train()
train_loss = 0
for minibatch in group_pairs_into_minibatches(train_data, batch_size, max_seq_length):
inputs, labels = zip(*[convert_transformer_inputs(tokenizer, sequence, max_seq_length) for sequence in minibatch])
inputs = tuple(item.to(device) for item in inputs)
labels = torch.cat([label.to(device) for label in labels], dim=0)
optimizer.zero_grad()
outputs = model(*inputs)
loss = outputs.loss
loss.backward()
optimizer.step()
scheduler.step()
train_loss += loss.item()

model.eval()
val_loss = 0
val_predictions = []
val_true_labels = []
for minibatch in group_pairs_into_minibatches(val_data, batch_size, max_seq_length):
inputs, labels = zip(*[convert_transformer_inputs(tokenizer, sequence, max_seq_length) for sequence in minibatch])
inputs = tuple(item.to(device) for item in inputs)
labels = torch.cat([label.to(device) for label in labels], dim=0)
with torch.no_grad():
outputs = model(*inputs)
val_loss += outputs.loss.item()
logits, probs, predictions = convert_transformer_outputs(outputs, num_tags)
val_predictions.extend(predictions.cpu().numpy().tolist())
val_true_labels.extend(labels.cpu().numpy().tolist())

print(f"Epoch {epoch + 1}/{num_epochs}")
print(f"Train loss: {train_loss / len(train_data)}")
print(f"Val loss: {val_loss / len(val_data)}")
print(classification_report([f"{i}" for i in val_true_labels], val_predictions, target_names=[f"TAG_{i}" for i in range(num_tags)]))


if __name__ == "__main__":
main()