from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass
import torch

@dataclass
class BatchEncoding:
> input\_ids: torch.Tensor
> attention\_mask: torch.Tensor
> labels: torch.Tensor

def create\_transformer\_tagger\_model(num\_labels):
model = AutoModelForTokenClassification.from\_pretrained("bert-base-cased", num\_labels=num\_labels)
return model

def create\_transformer\_tokenizer(vocab\_file, merges\_file):
tokenizer = AutoTokenizer.from\_pretrained(vocab\_file, merges\_file)
return tokenizer

def convert\_transformer\_inputs(tokenizer, text):
encoded = tokenizer(text, return\_tensors="pt", padding="max\_length", truncation=True, max\_length=512)
return encoded.input\_ids, encoded.attention\_mask

def convert\_transformer\_outputs(model\_output, labels):
predictions = model\_output.logits
return predictions, labels

def evaluate\_sequences(predictions, labels):
return torch.argmax(predictions, dim=-1) == labels

def group\_pairs\_into\_minibatches(pair\_list, batch\_size):
return [pair\_list[i:i + batch\_size] for i in range(0, len(pair\_list), batch\_size)]

def main():
if torch.cuda.is\_available():
device = "cuda"
else:
device = "cpu"

config = {
'model\_name': "bert-base-cased",
'num\_labels': 2,
'batch\_size': 16,
'epochs': 3,
'learning\_rate': 5e-5,
'warmup\_steps': 500,
'weight\_decay': 0.01,
'save\_steps': 1000,
'logging\_steps': 100,
'evaluation\_strategy': "epoch",
'load\_best\_model\_at\_end': True,
'metric\_for\_best\_model': "accuracy",
'greater\_is\_better': True,
'label\_smoothing\_factor': 0.1,
'gradient\_accumulation\_steps': 1,
'fp16': False,
'fp16\_opt\_level': "O1",
'movement\_decay': 0.99,
'n\_smoothing': 1,
'dynamic\_loss\_scaling': False,
'adafactor': False,
'output\_dir': "./results",
'overwrite\_output\_dir': True,
'logging\_dir': "./logs",
'mixed\_precision': "no",
'run\_name': "test\_run",
'save\_total\_limit': 2,
'save\_strategy': "steps",
'datasets\_name': "cased\_ Wikipedia",
'dataset\_config\_name': None,
'train\_file': "train.txt",
'validation\_file': "validation.txt",
'test\_file': "test.txt",
'per\_device\_train\_batch\_size': 16,
'per\_device\_eval\_batch\_size': 16,
'gradient\_accumulation\_steps': 1,
'gradient\_checkpointing': False,
'preprocessing\_num\_workers': 4,
'load\_dataset': "tensorflow",
'save\_strategy': "steps",
'save\_steps': 1000,
'save\_total\_limit': 2,
'save\_on\_each\_node': False,
'resume\_from\_checkpoint': None,
'fp16': False,
'fp16\_opt\_level': "O1",
'fp16\_opt\_level\_test': None,
'fp16\_opt\_shard\_size\_bytes': 512 \* 1024 \* 1024,
'fp16\_on\_cpu': False,
'fp16\_full\_eval': False,
'log\_level': "info",
'log\_on\_each\_node': True,
'log\_batch': 100,
'log\_step': 100,
'logging\_nan\_inf\_inf\_filter': "drop",
'logging\_steps': 100,
'load\_best\_model\_at\_end': True,
'metric\_for\_best\_model': "accuracy",
'greater\_is\_better': True,
'fast\_dev\_run': False,
'eval\_steps': None,
'eval\_accumulation\_steps': 1,
'eval\_batch\_size': 16,
'eval\_shard\_size\_bytes': 512 \* 1024 \* 1024,
'eval\_on': "cpu",
'dynamic\_loss\_scaling': False,
'dynamic\_loss\_scaling\_smoothing\_factor': 0.1,
'dynamic\_loss\_scaling\_max\_loss\_scale': 128,
'dynamic\_loss\_scaling\_min\_loss\_scale': 1,
'dynamic\_loss\_scaling\_loss\_scale\_window': 100,
'dynamic\_loss\_scaling\_gradient\_clipping': 1.0,
'adam\_beta1': 0.9,
'adam\_beta2': 0.999,
'adam\_epsilon': 1e-08,
'max\_grad\_norm': 1.0,
'max\_steps': -1,
'max\_time': "00:00:00",
'gradient\_clip\_val': 0.0,
'gradient\_clip\_algorithm': "value",
'gradient\_clipping': 0.0,
'gradient\_accumulation\_steps': 1,
'gradient\_checkpointing': False,
'gradient\_checkpointing\_num\_layers': 3,
'gradient\_checkpointing\_keep\_every\_n\_steps': 5,
'gradient\_checkpointing\_deepcopy': False,
'gradient\_checkpointing\_shard\_size\_bytes': 512 \* 1024 \* 1024,
'gradient\_checkpointing\_dropout': 0.0,
'gradient\_checkpointing\_dropout\_add': 0.0,
'gradient\_checkpointing\_dropout\_keep': 0.0,
'gradient\_checkpointing\_dropout\_mult': 0.0,
'gradient\_checkpointing\_dropout\_rng\_seed': 42,
'gradient\_checkpointing\_dropout\_rng\_state\_seed': 42,
'gradient\_checkpointing\_dropout\_rng\_state\_keep': 0,
'gradient\_checkpointing\_dropout\_rng\_state\_keep\_for\_newelt': 0,
'gradient\_checkpointing\_dropout\_rng\_state\_keep\_for\_newelt\_add': 0,
'gradient\_checkpointing\_dropout\_rng\_state\_keep\_for\_newelt\_mult': 0,
'gradient\_checkpointing\_dropout\_rng\_state\_keep\_for\_newelt\_keep': 0,
'gradient\_checkpointing\_dropout\_rng\_state\_keep\_for\_newelt\_keep\_add': 0,
'gradient\_checkpointing\_dropout\_rng\_state\_keep\_for\_newelt\_keep\_mult': 0,
'gradient\_checkpointing\_dropout\_rng\_state\_keep\_for\_newelt\_keep\_keep': 0,
'gradient\_checkpointing\_dropout\_rng\_state\_keep\_for\_newelt\_keep\_keep\_add': 0,
'gradient\_checkpointing\_dropout\_rng\_state\_keep\_for\_newelt\_keep\_keep\_mult': 0,
'gradient\_checkpointing\_dropout\_rng\_state\_keep\_for\_newelt\_keep\_keep\_keep': 0,
'gradient\_checkpointing\_dropout\_rng\_state\_keep\_for\_newelt\_keep\_keep\_keep\_add': 0,
'gradient\_checkpointing\_dropout\_rng\_state\_keep\_for\_newelt\_keep\_keep\_keep\_mult': 0,
'gradient\_checkpointing\_drop