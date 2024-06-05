 Here is the Python code that follows your instructions, using the Avalanche library and PyTorch:

```python
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from avalanche.benchmarks import MNISTBenchmark
from avalanche.model_wrappers import PreTrainedModel
from avalanche.optimizers import SGD
from avalanche.losses import CrossEntropyLoss
from avalanche.evaluation.plugins import Accuracy, ClassAccuracy, AMCA, Forgetting, BackwardTransfer, ForwardTransfer, CPUUsage, Timing, RAMUsage, GPUUsage, DiskUsage, MAC, LabelsRepartition
from avalanche.utils.logger import TextLogger, InteractiveLogger, CSVLogger, TensorboardLogger
from avalanche.utils.initializers import XavierUniform
from avalanche.utils.experiment import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
device = torch.device(args.device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

benchmark = MNISTBenchmark(data_dir='./data', num_classes=10, train_transform=transform, test_transform=transform)

class SimpleMLP(nn.Module):
    def __init__(self, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc3.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleMLP(benchmark.num_classes).to(device)

optimizer = SGD(model.parameters(), lr=0.01)
loss_fn = CrossEntropyLoss()

evaluation_plugins = [
    Accuracy(),
    ClassAccuracy(),
    AMCA(),
    Forgetting(),
    BackwardTransfer(),
    ForwardTransfer(),
    CPUUsage(),
    Timing(),
    RAMUsage(),
    GPUUsage(),
    DiskUsage(),
    MAC(),
    LabelsRepartition()
]

strategy = PreTrainedModel(model, optimizer, loss_fn, evaluation_plugins=evaluation_plugins)

loggers = [TextLogger(), InteractiveLogger(), CSVLogger('./logs.csv'), TensorboardLogger('./logs')]

run_experiment(
    benchmark=benchmark,
    strategy=strategy,
    num_epochs=10,
    num_tasks=10,
    num_experiences=1,
    loggers=loggers,
    device=device
)

for plugin in evaluation_plugins:
    print(plugin.metrics)
```

This code assumes that the MNIST dataset is located in the './data' directory. Make sure to install the Avalanche library and PyTorch before running the code.