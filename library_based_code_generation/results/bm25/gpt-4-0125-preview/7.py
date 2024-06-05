```python
import argparse
from torchvision import transforms
from avalanche.benchmarks.classic import MNIST
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import TextLogger, InteractiveLogger, CSVLogger, TensorboardLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, class_accuracy_metrics, \
    forgetting_metrics, backward_transfer, forward_transfer_metrics, cpu_usage_metrics, timing_metrics, \
    ram_usage_metrics, gpu_usage_metrics, disk_usage_metrics, MAC_metrics, labels_repartition_metrics
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

# Parse command line arguments
parser = argparse.ArgumentParser(description='Avalanche MNIST Example')
parser.add_argument('--device', type=str, default='cpu', help='Device to use')
args = parser.parse_args()

# Define transformations for training and testing data
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Create a benchmark
benchmark = MNIST(train_transform=train_transform, eval_transform=test_transform)

# Create a simple MLP model
model = SimpleMLP(num_classes=benchmark.n_classes)

# Define loggers
text_logger = TextLogger()
interactive_logger = InteractiveLogger()
csv_logger = CSVLogger('logs')
tensorboard_logger = TensorboardLogger('tensorboard_logs')

# Define an evaluation plugin
eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    class_accuracy_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    backward_transfer(experience=True),
    forward_transfer_metrics(stream=True),
    cpu_usage_metrics(experience=True, epoch=True, stream=True),
    timing_metrics(epoch=True, experience=True, stream=True),
    ram_usage_metrics(experience=True, epoch=True, stream=True),
    gpu_usage_metrics(experience=True, epoch=True, stream=True),
    disk_usage_metrics(experience=True, epoch=True, stream=True),
    MAC_metrics(epoch=True, experience=True, stream=True),
    labels_repartition_metrics(epoch=True, experience=True, stream=True),
    loggers=[text_logger, interactive_logger, csv_logger, tensorboard_logger]
)

# Create a Naive continual learning strategy
strategy = Naive(
    model=model,
    optimizer=SGD(model.parameters(), lr=0.001, momentum=0.9),
    criterion=CrossEntropyLoss(),
    train_mb_size=100,
    train_epochs=1,
    eval_mb_size=100,
    device=args.device,
    plugins=[eval_plugin]
)

# Train and evaluate the model
for experience in benchmark.train_stream:
    print(f"Training on experience {experience.current_experience}")
    strategy.train(experience)
    print("Training completed")

    print(f"Evaluating on experience {experience.current_experience}")
    res = strategy.eval(experience)
    print("Evaluation completed")
    print(res)

# After all experiences
print("Final evaluation on test stream")
for experience in benchmark.test_stream:
    res = strategy.eval(experience)
    print(res)
```