```python
import torch
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
from torchvision import transforms
from torchvision.models import resnet18

def main():
    # Hyperparameters
    lr = 0.01
    momentum = 0.9
    batch_size = 128
    epochs = 30
    log_every = 1
    seed = 1234
    torch.manual_seed(seed)

    # Model
    model = resnet18(pretrained=False, num_classes=10)

    # Normalization and Transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Benchmark
    benchmark = SplitCIFAR10(n_experiences=5, return_task_id=False, seed=seed, train_transform=transform, eval_transform=transform)

    # Logging
    text_logger = TextLogger(open('log.txt', 'a'))
    tensorboard_logger = TensorboardLogger()
    interactive_logger = InteractiveLogger()

    # Evaluation Plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tensorboard_logger]
    )

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and Scheduler
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    lr_scheduler_plugin = LRSchedulerPlugin(lr_scheduler)

    # Continual Learning Strategy
    strategy = Naive(
        model, optimizer, torch.nn.CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=epochs, eval_mb_size=100,
        device=device, plugins=[lr_scheduler_plugin], evaluator=eval_plugin
    )

    # Training Loop
    for experience in benchmark.train_stream:
        strategy.train(experience)
        strategy.eval(benchmark.test_stream)

if __name__ == "__main__":
    main()
```