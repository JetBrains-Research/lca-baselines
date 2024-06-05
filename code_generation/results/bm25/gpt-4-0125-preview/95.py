```python
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from avalanche.benchmarks import GenericCLScenario
from avalanche.models import Naive
from avalanche.training.strategies import Naive as NaiveStrategy
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

class T5ForQuestionAnswering(Naive):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids, labels=labels)

def preprocess_data(dataset):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    def preprocess_function(examples):
        inputs = ["question: " + q + " context: " + c for q, c in zip(examples['question'], examples['context'])]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['answers']['text'][0], max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    dataset = dataset.map(preprocess_function, batched=True)
    return dataset

def main():
    dataset = load_dataset("squad")
    train_dataset = preprocess_data(dataset["train"])
    val_dataset = preprocess_data(dataset["validation"])

    model_name = "t5-small"
    model = T5ForQuestionAnswering(model_name)

    train_scenario = GenericCLScenario(train_dataset, task_labels=[0 for _ in range(len(train_dataset))], complete_test_set_only=False)
    val_scenario = GenericCLScenario(val_dataset, task_labels=[0 for _ in range(len(val_dataset))], complete_test_set_only=True)

    logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(accuracy_metrics(epoch=True, experience=True),
                                   loss_metrics(epoch=True, experience=True),
                                   logger=logger)

    strategy = NaiveStrategy(model,
                             torch.optim.Adam(model.parameters(), lr=5e-5),
                             train_mb_size=16,
                             train_epochs=1,
                             eval_mb_size=16,
                             device="cuda" if torch.cuda.is_available() else "cpu",
                             plugins=[eval_plugin])

    for experience in train_scenario.train_stream:
        strategy.train(experience)
        strategy.eval(val_scenario.test_stream)

    # Example question
    question = "What is the capital of France?"
    context = "France is a country in Europe. The capital of France is Paris."

    input_text = "question: " + question + " context: " + context
    input_ids = model.tokenizer(input_text, return_tensors="pt").input_ids

    # Generate answer
    outputs = model.model.generate(input_ids)
    answer = model.tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
```