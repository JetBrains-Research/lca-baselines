 Here is the Python code that follows your instructions using the avalanche library:

```python
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from avalanche.benchmarks import MNIST
from avalanche.dataset import StreamCPUUsage, StreamDiskUsage
from avalanche.metrics import accuracy_metrics, class_accuracy_metrics, loss_metrics, timing_metrics, forward_transfer_metrics, AMCA, Forgetting, BackwardTransfer
from avalanche.model_wrappers import ClassifierWrapper
from avalanche.optimizers import SGD
from avalanche.plugins import EvaluationPlugin, EvaluationPluginTest
from avalanche.utils.metrics import MAC
from avalanche.utils.torch_utils import cpu_usage_metrics, gpu_usage_metrics, disk_usage_metrics, log_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()
device = torch.device(args.device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_benchmark = MNIST(root='./data', train=True, transform=transform, download=True)
test_benchmark = MNIST(root='./data', train=False, transform=transform, download=True)

train_stream = train_benchmark.to_stream(StreamCPUUsage(), StreamDiskUsage())
test_stream = test_benchmark.to_stream(StreamCPUUsage(), StreamDiskUsage())

model = ClassifierWrapper(nn.Sequential(
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
), num_classes=10)

optimizer = SGD(model.parameters(), lr=0.01)

evaluation_plugin = EvaluationPlugin(
    accuracy=accuracy_metrics,
    class_accuracy=class_accuracy_metrics,
    loss=loss_metrics,
    forward_transfer=forward_transfer_metrics,
    AMCA=AMCA,
    forgetting=Forgetting,
    backward_transfer=BackwardTransfer,
    CPU_usage=cpu_usage_metrics,
    timing=timing_metrics,
    MAC=MAC,
    labels_repartition=MAC
)

evaluation_plugin_test = EvaluationPluginTest(
    accuracy=accuracy_metrics,
    class_accuracy=class_accuracy_metrics,
    loss=loss_metrics,
    forward_transfer=forward_transfer_metrics,
    AMCA=AMCA,
    forgetting=Forgetting,
    backward_transfer=BackwardTransfer,
    CPU_usage=cpu_usage_metrics,
    timing=timing_metrics,
    MAC=MAC,
    labels_repartition=MAC
)

check_model_and_optimizer(model, optimizer)

strategy = None  # Naive continual learning strategy

for experience in train_stream:
    model, optimizer = strategy.learn_one_shot(model, optimizer, experience.data, experience.targets)
    evaluation_plugin._update_metrics_and_loggers(model, experience.data, experience.targets)
    print("Experience:", experience.index, "Accuracy:", evaluation_plugin.accuracy.value)

    if experience.index % 10 == 0:
        evaluation_plugin_test._update_metrics_and_loggers(model, test_stream.next_batch())
        log_metrics(evaluation_plugin_test, device)

log_metrics(evaluation_plugin, device)
```

This code assumes that you have the CelebA dataset in the './data' directory. You may need to adjust the paths according to your setup. Also, the code does not include the definition of the Naive continual learning strategy, as it was not specified in your instructions. You can define it according to your requirements.