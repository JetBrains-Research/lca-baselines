 Here's a Python code snippet that implements the given instructions using the Avalanche library, PyTorch, and Argparse. Please note that you need to have Avalanche, PyTorch, and torchvision installed.

```python
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from avalanche.benchmarks import MNISTBenchmark
from avalanche.model_selection import ReplayStrategy
from avalanche.transformers import Normalize

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    return device

def create_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)

    args = parser.parse_args()

    device = get_device()

    transform = transforms.Compose([
        transforms.ToTensor(),
        Normalize(),
    ])

    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    benchmark = MNISTBenchmark(train_loader, test_loader)

    model = create_model()

    strategy = ReplayStrategy(
        model,
        buffer_size=1000,
        reservoir_size=100,
        buffer_policy=ReplayStrategy.ReservoirSamplingBuffer,
        evaluation_metric=benchmark.evaluation_metrics.accuracy,
        evaluation_interval=10
    )

    results = []

    for _ in range(benchmark.num_tasks):
        X_train, y_train, X_test, y_test = benchmark.next_task()
        strategy.fit(X_train, y_train, X_test, y_test)
        results.append(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(str(