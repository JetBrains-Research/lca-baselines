```python
import thinc
from thinc.api import chain, Adam, Config, fix_random_seed, prefer_gpu
from thinc.types import Ragged, Floats2d
from thinc.model import Model
from thinc.layers import with_array, PyTorchWrapper
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from typing import List, Tuple, Callable, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

@dataclass
class EncodedData:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]

def create_transformer_tokenizer(model_name: str) -> Callable:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def create_transformer_model(model_name: str) -> Model:
    transformer_model = AutoModelForTokenClassification.from_pretrained(model_name)
    return PyTorchWrapper(transformer_model)

def convert_inputs(tokenizer: Callable, texts: List[str], tags: List[List[int]], max_length: int) -> List[EncodedData]:
    encoded_texts = tokenizer.batch_encode_plus(texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    encoded_data = []
    for i in range(len(texts)):
        input_ids = encoded_texts['input_ids'][i].tolist()
        attention_mask = encoded_texts['attention_mask'][i].tolist()
        encoded_data.append(EncodedData(input_ids=input_ids, attention_mask=attention_mask, labels=tags[i]))
    return encoded_data

def convert_outputs(model_output) -> List[int]:
    logits = model_output.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.tolist()

def evaluate(model: Model, tokenizer: Callable, texts: List[str], true_tags: List[List[int]]) -> float:
    correct, total = 0, 0
    for text, true_tag in zip(texts, true_tags):
        encoded_data = convert_inputs(tokenizer, [text], [true_tag], max_length=512)[0]
        model_output = model.predict([encoded_data.input_ids, encoded_data.attention_mask])
        predictions = convert_outputs(model_output)
        correct += sum(p == t for p, t in zip(predictions, true_tag))
        total += len(true_tag)
    return correct / total

def minibatch(data: List[Any], size: int) -> List[List[Any]]:
    batches = [data[i:i + size] for i in range(0, len(data), size)]
    return batches

def main():
    config = Config().from_str("""
    [model]
    @layers = "transformer_tagger.v1"

    [optimizer]
    @optimizers = "Adam.v1"
    learn_rate = 0.001

    [training]
    n_epochs = 3
    batch_size = 8
    """)
    fix_random_seed(0)
    use_gpu = prefer_gpu()
    print(f"Using GPU: {use_gpu}")

    model_name = "bert-base-uncased"
    tokenizer = create_transformer_tokenizer(model_name)
    transformer_model = create_transformer_model(model_name)

    # Example dataset
    texts = ["Hello, world!", "Thinc and Transformers"]
    tags = [[1, 0], [2, 0, 0, 0]]

    encoded_data = convert_inputs(tokenizer, texts, tags, max_length=512)
    batches = minibatch(encoded_data, config["training"]["batch_size"])

    optimizer = Adam(lr=config["optimizer"]["learn_rate"])

    for epoch in range(config["training"]["n_epochs"]):
        losses = []
        for batch in tqdm(batches, desc=f"Training Epoch {epoch+1}"):
            X = [torch.tensor([example.input_ids for example in batch]), torch.tensor([example.attention_mask for example in batch])]
            Y = torch.tensor([example.labels for example in batch])
            loss, backprop = transformer_model.begin_update(X)
            backprop(Y)
            optimizer(transformer_model, X)
            losses.append(loss)
        print(f"Epoch {epoch+1}: Loss = {np.mean(losses)}")

    accuracy = evaluate(transformer_model, tokenizer, texts, tags)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
```