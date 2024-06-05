```python
import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

# Argument parser for CUDA device selection
parser = argparse.ArgumentParser(description='Online Continual Learning with Avalanche')
parser.add_argument('--cuda', type=int, default=0, help='CUDA device number')
args = parser.parse_args()

# Set the device
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# Configurations and transformations
scenario = SplitMNIST(n_experiences=5, return_task_id=False, fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Model
model = SimpleMLP(num_classes=10)

# Strategy
strategy = Naive(
    model=model,
    optimizer=SGD(model.parameters(), lr=0.001, momentum=0.9),
    criterion=CrossEntropyLoss(),
    train_mb_size=100,
    train_epochs=1,
    eval_mb_size=100,
    device=device,
    plugins=[ReplayPlugin(mem_size=1000, storage_policy=ReservoirSamplingBuffer())],
    evaluator=EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger()]
    )
)

# Training and evaluation loop
results = []
for experience in scenario.train_stream:
    strategy.train(experience)
    print('Training completed')

    print('Starting evaluation')
    res = strategy.eval(scenario.test_stream)
    results.append(res)
```