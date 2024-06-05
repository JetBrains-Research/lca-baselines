import torch
from avalanche import training, evaluation, strategies
from avalanche.benchmarks import MNIST
from avalanche.models import SimpleMLP
from avalanche.training.plugins import ReplayPlugin
from avalanche.storage import ReservoirSamplingBuffer
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
import argparse

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='Select CUDA device to use')
args = parser.parse_args()

# Set up configurations and transformations
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
scenario = MNIST(n_experiences=10, task_labels=False)
model = SimpleMLP(num_classes=scenario.n_classes, input_size=28*28).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
metrics = accuracy_metrics()

# Create Replay strategy instance
replay_strategy = strategies.Replay(
    model, optimizer, criterion, ReservoirSamplingBuffer(max_size=2000),
    train_mb_size=10, train_epochs=1, eval_mb_size=10, device=device,
    evaluator=evaluation.Evaluator(metrics=metrics),
    train_metrics=metrics, eval_every=100, loggers=[InteractiveLogger()]
)

# Training loop
results = []
for experience in scenario.train_stream:
    replay_strategy.train(experience)
    results.append(replay_strategy.eval(scenario.test_stream))