  import argparse
import torch
from torchvision import datasets, transforms
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.strategies import Replay
from avalanche.training.storage_policies import ReservoirSamplingBuffer

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='GPU device ID')
args = parser.parse_args()

# set up configurations
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

# set up transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# set up scenario
scenario = avalanche.benchmarks.scenarios.ContinualScenario(
    avalanche.benchmarks.datasets.MNIST(root='./data', download=True, transform=transform),
    avalanche.benchmarks.scenarios.ClassIncremental(
        increment=10,
        max_classes_increment=10,
        shuffle_train=True,
        shuffle_test=False,
        reset_targets=False
    )
)

# set up model
model = SimpleMLP(input_size=784, hidden_size=128, num_classes=10)

# set up evaluation metrics
metrics = [accuracy_metrics.Accuracy(), accuracy_metrics.Error()]

# set up Replay strategy
replay_strategy = Replay(
    model,
    scenario,
    ReservoirSamplingBuffer(1000),
    metrics,
    device=device
)

# set up training loop
results = []
for experience in scenario.train_stream:
    replay_strategy.train(experience)
    results.append(replay_strategy.eval(scenario.test_stream))

# print results
print(results)