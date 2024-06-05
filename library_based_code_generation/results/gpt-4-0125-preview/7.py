```python
import argparse
import torch
from torchvision import transforms
from avalanche.benchmarks.classic import MNIST
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.evaluation.metrics import *
from avalanche.logging import InteractiveLogger, TextLogger, CSVLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin

# 1. Parse command line arguments
parser = argparse.ArgumentParser(description='Avalanche MNIST Example')
parser.add_argument('--device', type=str, default='cpu', help='Device to use')
args = parser.parse_args()

# 2. Define transformations
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 3. Create a benchmark
benchmark = MNIST(train_transform=train_transform, eval_transform=eval_transform)

# 4. Create a simple MLP model
model = SimpleMLP(num_classes=benchmark.n_classes)

# 5. Define loggers
text_logger = TextLogger()
interactive_logger = InteractiveLogger()
csv_logger = CSVLogger(filename='log.csv')
tensorboard_logger = TensorboardLogger(tb_log_dir='tb_data')

# 6. Define an evaluation plugin
eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    class_accuracy_metrics(epoch=True, experience=True, stream=True),
    AMCA_metric(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    backward_transfer_metrics(stream=True),
    forward_transfer_metrics(stream=True),
    cpu_usage_metrics(experience=True, stream=True),
    timing_metrics(epoch=True, experience=True, stream=True),
    ram_usage_metrics(experience=True, stream=True),
    gpu_usage_metrics(experience=True, stream=True),
    disk_usage_metrics(experience=True, stream=True),
    MAC_metrics(epoch=True, experience=True, stream=True),
    labels_repartition_metrics(epoch=True, experience=True, stream=True),
    loggers=[text_logger, interactive_logger, csv_logger, tensorboard_logger]
)

# 7. Create a Naive continual learning strategy
strategy = Naive(
    model=model,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.001),
    criterion=torch.nn.CrossEntropyLoss(),
    train_mb_size=100,
    train_epochs=1,
    eval_mb_size=100,
    device=args.device,
    plugins=[eval_plugin]
)

# 8. Train and evaluate the model
for experience in benchmark.train_stream:
    strategy.train(experience)
    strategy.eval(benchmark.test_stream)

# 9. Print all metrics
print(eval_plugin.get_all_metrics())
```