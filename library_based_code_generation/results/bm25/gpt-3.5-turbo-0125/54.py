```python
import argparse
import torch
from avalanche import create_multi_dataset_generic_scenario
from avalanche.benchmarks import MNIST
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Replay
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.models.dynamic import MultiTaskModule
from avalanche.training.plugins import ReplayPlugin
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.training.strategies.replay.replay_strategy import ReservoirSamplingBuffer

# Set up configurations and transformations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnist_train = MNIST(train=True)
mnist_test = MNIST(train=False)
scenario = create_multi_dataset_generic_scenario(
    train_datasets=mnist_train,
    test_datasets=mnist_test,
    n_experiences=1,
    task_labels=False
)

# Create SimpleMLP model
model = SimpleMLP(num_classes=scenario.n_classes)

# Choose evaluation metrics
metrics = accuracy_metrics()

# Create Replay strategy with ReservoirSamplingBuffer storage policy
replay_strategy = Replay(
    model,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.001),
    criterion=torch.nn.CrossEntropyLoss(),
    train_mb_size=10,
    train_epochs=1,
    eval_mb_size=10,
    device=device,
    evaluator=EvaluationPlugin(
        metrics=metrics,
        loggers=[InteractiveLogger()]
    ),
    buffer=ReservoirSamplingBuffer(max_size=1000)
)

# Training loop
results = []
for experience in scenario.train_stream:
    replay_strategy.train(experience)
    results.append(replay_strategy.eval(scenario.test_stream))

```