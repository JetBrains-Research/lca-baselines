import argparse
from avalanche.benchmarks import MNIST
from avalanche.models import SimpleMLP
from avalanche.logging import TextLogger, InteractiveLogger, CSVLogger, TensorboardLogger
from avalanche.evaluation import PluginMetric, accuracy_metrics, loss_metrics, class_accuracy_metrics, AMCA, forgetting_metrics, backward_transfer_metrics, forward_transfer_metrics, CPUUsage, timing_metrics, RAMUsage, GPUUsage, disk_usage_metrics, MAC, labels_repartition_metrics
from avalanche.training.strategies import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.optim import SGD
from avalanche.losses import CrossEntropyLoss

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

# Define transformations for training and testing data
# (Assuming transformations are defined elsewhere)

# Create a benchmark using the MNIST dataset with the defined transformations
benchmark = MNIST(n_experiences=10, train_transformations=..., test_transformations=...)

# Create a simple MLP model
model = SimpleMLP(input_size=28*28, num_classes=benchmark.n_classes)

# Define loggers
text_logger = TextLogger()
interactive_logger = InteractiveLogger()
csv_logger = CSVLogger()
tensorboard_logger = TensorboardLogger()

# Define evaluation plugin
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    class_accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    AMCA(),
    forgetting_metrics(),
    backward_transfer_metrics(),
    forward_transfer_metrics(),
    CPUUsage(),
    timing_metrics(),
    RAMUsage(),
    GPUUsage(),
    disk_usage_metrics(),
    MAC(),
    labels_repartition_metrics(),
    loggers=[text_logger, interactive_logger, csv_logger, tensorboard_logger]
)

# Create a Naive continual learning strategy
strategy = Naive(
    model, SGD(), CrossEntropyLoss(), eval_plugin
)

# Train and evaluate the model
for experience in benchmark.train_stream:
    strategy.train(experience)
    results = strategy.eval(benchmark.test_stream)
    print(results)

# Print all metrics stored by the evaluation plugin
print(eval_plugin.get_all_metrics())