 import argparse
import avalanche as av
from avalanche.benchmarks import MNIST
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.evaluation.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    AverageMetric,
    TimeElapsed,
    ClassAccuracy,
    Loss,
    MAC,
    AMCA,
    Forgetting,
    BackwardTransfer,
    ForwardTransfer,
    LabelsRepartition,
    DiskUsageMetrics,
    CPUUsageMetrics,
    GPUUsageMetrics,
    RAMUsageMetrics,
)

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

transforms = av.transforms.Compose([
    av.transforms.ToTensor(),
    av.transforms.Normalize(mean=[0.1307], std=[0.3081]),
])

benchmark = MNIST(transforms=transforms)

model = SimpleMLP(
    num_classes=benchmark.num_classes,
    hidden_size=128,
    output_size=benchmark.num_classes,
)

optimizer = av.optimizers.SGD(model.parameters(), lr=0.01)
criterion = av.losses.CrossEntropyLoss()

evaluation_plugin = EvaluationPlugin(
    metrics=[
        AverageMetric(TimeElapsed()),
        AverageMetric(ClassAccuracy()),
        AverageMetric(Loss(criterion)),
        AverageMetric(MAC()),
        AverageMetric(AMCA()),
        AverageMetric(Forgetting()),
        AverageMetric(BackwardTransfer()),
        AverageMetric(ForwardTransfer()),
        AverageMetric(LabelsRepartition()),
        DiskUsageMetrics(),
        CPUUsageMetrics(),
        GPUUsageMetrics(),
        RAMUsageMetrics(),
    ],
    loggers=[
        av.logging.TextLogger(),
        av.logging.InteractiveLogger(),
        av.logging.CSVLogger(),
        av.logging.TensorboardLogger(),
    ],
)

strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    eval_plugin=evaluation_plugin,
)

strategy.fit(
    benchmark.train_stream,
    n_epochs=1,
    validate=True,
    verbose=True,
)

strategy.evaluate(benchmark.test_stream)

print(evaluation_plugin.get_metrics())