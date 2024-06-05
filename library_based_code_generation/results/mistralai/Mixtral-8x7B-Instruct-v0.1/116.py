 ```python
import os
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from avalanche.benchmarks import CLEARBenchmark
from avalanche.models import ResNet18
from avalanche.training.strategies import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import TensorboardLogger, CSVLogger, VerboseLogger

# Hyperparameters
num_workers = 4
batch_size = 128
lr = 0.1
momentum = 0.9
weight_decay = 1e-4
num_epochs = 10
log_interval = 10

# Learning rate scheduler
def learning_rate_scheduler(optimizer, epoch):
    return StepLR(optimizer, step_size=1, gamma=0.1)

def main():
    # Initialize model
    model = ResNet18(num_classes=10)

    # Define normalization and transformation operations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize,
    ])

    # Set up logging
    tb_logger = TensorboardLogger(save_dir="tb_logs", name="clear_run")
    csv_logger = CSVLogger(save_dir="logs", name="clear_run")
    verbose_logger = VerboseLogger(log_interval=log_interval)

    # Define evaluation plugin
    evaluation_plugin = EvaluationPlugin(
        metrics={
            "accuracy": True,
            "top5_accuracy": True,
            "precision": True,
            "recall": True,
            "f1_score": True,
        },
        log_metrics_every_n_steps=10,
        logger=verbose_logger,
    )

    # Set seed value and create CLEAR benchmark
    seed_value = 42
    torch.manual_seed(seed_value)
    clear_benchmark = CLEARBenchmark(
        dataset="CIFAR100",
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        seed_experiment=seed_value,
        seed_tasks=seed_value,
        evaluation_strategy="task",
        stream_sequential=True,
    )

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define SGD optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    learning_rate_scheduler_fn = learning_rate_scheduler

    # Define continual learning strategy
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        scheduler=learning_rate_scheduler_fn,
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=batch_size,
        valid_mb_size=batch_size,
        train_epochs=num_epochs,
        eval_epochs=num_epochs,
        plugins=[evaluation_plugin, tb_logger, csv_logger, verbose_logger],
    )

    # Run training loop
    strategy.fit(clear_benchmark.train_stream, clear_benchmark.test_stream, n_experiences=clear_benchmark.n_experiences)

    # Generate accuracy matrix and compute CLEAR metrics
    accuracy_matrix = evaluation_plugin.get_metric("confusion_matrix")
    clear_metrics = clear_benchmark.compute_metrics(accuracy_matrix)

    # Log CLEAR metrics to text file
    with open("clear_metrics.txt", "w") as f:
        f.write(str(clear_metrics))

if __name__ == "__main__":
    main()
```