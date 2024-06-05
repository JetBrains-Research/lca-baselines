```python
import torch
from dataclasses import dataclass
from transformers import AutoModelForTokenClassification, AutoTokenizer
from thinc.api import Config, chain, Model
from thinc.types import Ragged, Floats2d
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

@dataclass
class EncodedData:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]

def create_transformer_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)

def create_transformer_model(model_name: str) -> AutoModelForTokenClassification:
    return AutoModelForTokenClassification.from_pretrained(model_name)

def create_transformer_tagger_model(transformer_model: AutoModelForTokenClassification) -> Model[List[EncodedData], List[Ragged]]:
    def forward(model, data: List[EncodedData], is_train: bool):
        # Implement forward pass
        pass
    return Model("transformer_tagger", forward)

def convert_transformer_inputs(batch: List[EncodedData]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([example.input_ids for example in batch])
    attention_mask = torch.tensor([example.attention_mask for example in batch])
    return input_ids, attention_mask

def convert_transformer_outputs(outputs: torch.Tensor) -> List[Ragged]:
    return [Ragged.from_numpy(output.numpy()) for output in outputs]

def evaluate_sequences(model: Model, data: List[EncodedData]) -> float:
    # Dummy evaluation function
    return 0.0

def group_into_minibatches(data: List[EncodedData], batch_size: int) -> List[List[EncodedData]]:
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def main():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load configuration
    config = Config().from_str("""
    [model]
    model_name = "bert-base-uncased"

    [optimizer]
    lr = 5e-5

    [training]
    epochs = 3
    batch_size = 32
    """)
    
    # Resolve configuration
    model_name = config["model"]["model_name"]
    lr = config["optimizer"]["lr"]
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    
    # Load dataset
    # This is a placeholder. Replace with actual dataset loading.
    dataset = []  # Should be a list of EncodedData instances
    
    # Initialize model and tokenizer
    tokenizer = create_transformer_tokenizer(model_name)
    transformer_model = create_transformer_model(model_name)
    model = create_transformer_tagger_model(transformer_model)
    
    # Training loop
    for epoch in range(epochs):
        losses = []
        for batch in tqdm(group_into_minibatches(dataset, batch_size), desc=f"Epoch {epoch}"):
            # Convert inputs
            input_ids, attention_mask = convert_transformer_inputs(batch)
            # Forward pass and compute loss
            # This is a placeholder. Implement actual forward pass and loss computation.
            loss = 0.0
            losses.append(loss)
        print(f"Epoch {epoch}: Average Loss = {np.mean(losses)}")
        
        # Evaluation (optional)
        eval_score = evaluate_sequences(model, dataset)
        print(f"Epoch {epoch}: Eval Score = {eval_score}")

if __name__ == "__main__":
    main()
```