```python
import argparse
from avalanche.benchmarks import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Replay
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.storage_policy import ReservoirSamplingBuffer
import torch

# Argument parser for CUDA device selection
parser = argparse.ArgumentParser(description='Select CUDA device')
parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device number')
args = parser.parse_args()

# Set device
device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

# Configurations and transformations
scenario = SplitMNIST(n_experiences=5, return_task_id=False, fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Model
model = SimpleMLP(num_classes=10)

# Choose metrics
eval_plugin = EvaluationPlugin(accuracy_metrics(epoch=True, experience=True, stream=True),
                               loss_metrics(epoch=True, experience=True, stream=True),
                               logger=InteractiveLogger())

# Replay strategy with ReservoirSamplingBuffer
strategy = Replay(model, torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
                  criterion=torch.nn.CrossEntropyLoss(), train_mb_size=100, train_epochs=1, eval_mb_size=100,
                  storage_policy=ReservoirSamplingBuffer(max_size=5000, adaptive_size=True),
                  evaluator=eval_plugin, device=device)

# Training and evaluation loop
results = []
for train_task in scenario.train_stream:
    print(f"Training on experience {train_task.current_experience}")
    strategy.train(train_task)
    print("Training completed")

    print("Evaluating on test stream")
    for test_task in scenario.test_stream:
        res = strategy.eval(test_task)
        results.append(res)
```