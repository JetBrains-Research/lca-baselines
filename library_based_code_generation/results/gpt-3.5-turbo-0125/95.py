```python
from avalanche.benchmarks import SQuAD
from avalanche.models import T5
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive
from transformers import T5Tokenizer
from transformers.avalanche import AvalancheT5

class CustomAvalancheT5(Naive):
    def __init__(self, model, optimizer, criterion, train_mb_size=32, eval_mb_size=32):
        super().__init__(model, optimizer, criterion, train_mb_size, eval_mb_size)

def main():
    squad = SQuAD()
    train_set, val_set = squad.get_dataset()

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5.from_pretrained('t5-base')
    model = AvalancheT5(model, tokenizer)

    scenario = CustomAvalancheT5(model, optimizer=None, criterion=None)
    scenario.train(train_set)
    
    question = "What is the capital of France?"
    answer = scenario.answer_question(question)
    print("Model's answer:", answer)

if __name__ == "__main__":
    main()
```