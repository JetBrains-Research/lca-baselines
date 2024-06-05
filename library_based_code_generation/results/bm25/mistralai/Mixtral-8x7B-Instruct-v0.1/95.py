 from avalanche.models import Naive
from avalanche.models.nlp import T5Model
from avalanche.training.templates import \
train_eval_template, supervised_template
from avalanche.training.strategies import \
CLSupervised, SCR
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import datasets

class SQuADModel(Naive):
def __init__(self, model_name: str, **kwargs):
super().__init__(**kwargs)
self.model = T5Model(AutoModelForQuestionAnswering.from_pretrained(model_name))
self.tokenizer = AutoTokenizer.from_pretrained(model_name)

def forward(self, inputs):
return self.model(**inputs)
def preprocess_fn(self, batch):
inputs = self.tokenizer(batch['question'], batch['context'], truncation=True, padding='longest', max_length=512, return_tensors="pt")
return inputs
def loss_fn(self, outputs, inputs):
return self.model.loss_fn(outputs, inputs)

def infer(self, inputs):
outputs = self.model(**inputs)
return self.model.infer(outputs)

def save_model(self, path):
self.model.save_pretrained(path)
self.tokenizer.save_pretrained(path)

def hash_model(self):
return hash(self.model.state_dict())

def get_model(path):
model = T5Model(AutoModelForQuestionAnswering.from_pretrained(path))
tokenizer = AutoTokenizer.from_pretrained(path)
return model, tokenizer

def main():
# Load SQuAD dataset
dataset = datasets.load_dataset('squad')

# Preprocess the dataset
preprocessed_dataset = dataset.map(SQuADModel.preprocess_fn, batched=True)

# Split the dataset into training and validation sets
train_set, valid_set = preprocessed_dataset.train_test_split(test_size=0.2)

# Initialize the T5 model
model, tokenizer = SQuADModel.get_model('t5-base')

# Set up the continual learning scenario with Avalanche
strategy = SCR(
CLSupervised(
model=model,
criterion=SQuADModel.loss_fn,
optimizer=torch.optim.AdamW,
lr=1e-5,
train_mb_size=4,
valid_mb_size=4,
train_epochs=1,
device='cuda',
)
)

# Train the model on the training set
strategy.fit(train_set, valid_set, num_epochs=1)

# Test the model by asking it a question
question = "Who was the first president of the United States?"
context = "George Washington was the first president of the United States."
inputs = tokenizer({'question': question, 'context': context}, return_tensors="pt")
answer = strategy.model.infer(strategy.model(**inputs))
print(answer)

if __name__ == '__main__':
main()