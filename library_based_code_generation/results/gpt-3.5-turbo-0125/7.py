import argparse
from avalanche.benchmarks import MNIST
from avalanche.logging import TextLogger, InteractiveLogger, CSVLogger, TensorboardLogger
from avalanche.evaluation import PluginMetric
from avalanche.training.strategies import Naive
from avalanche.models import SimpleMLP
from avalanche.optimizers import SGD
from avalanche.losses import CrossEntropyLoss

# 1. Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

# 2. Define transformations
train_transform = None
test_transform = None

# 3. Create benchmark
benchmark = MNIST(train_transform=train_transform, test_transform=test_transform)

# 4. Create simple MLP model
model = SimpleMLP(input_size=28*28, num_classes=benchmark.n_classes)

# 5. Define loggers
text_logger = TextLogger()
interactive_logger = InteractiveLogger()
csv_logger = CSVLogger()
tensorboard_logger = TensorboardLogger()

# 6. Define evaluation plugin
metrics = [PluginMetric.ACCURACY, PluginMetric.LOSS, PluginMetric.CLASS_ACCURACY, PluginMetric.AMCA,
           PluginMetric.FORGETTING, PluginMetric.BACKWARD_TRANSFER, PluginMetric.FORWARD_TRANSFER,
           PluginMetric.CPU_USAGE, PluginMetric.TIMING, PluginMetric.RAM_USAGE, PluginMetric.GPU_USAGE,
           PluginMetric.DISK_USAGE, PluginMetric.MAC, PluginMetric.LABELS_REPARTITION]
evaluation_plugin = PluginMetric(metrics, loggers=[text_logger, interactive_logger, csv_logger, tensorboard_logger])

# 7. Create Naive continual learning strategy
strategy = Naive(model, SGD(model.parameters(), lr=0.001), CrossEntropyLoss(), evaluation_plugin)

# 8. Train and evaluate model
for experience in benchmark.train_stream:
    strategy.train(experience)
    strategy.eval(experience, stream=benchmark.test_stream)
    print(strategy.evaluator.log())

# 9. Print all metrics
print(evaluation_plugin.get_all_metrics())