  import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import avalanche as avl
import avalanche.benchmarks as benchmarks
import avalanche.evaluation as evaluation
import avalanche.logging as logging
import avalanche.models as models
import avalanche.training as training
import avalanche.training.plugins as plugins
import avalanche.evaluation.metrics as metrics
import numpy as np
import os

# Define hyperparameters
num_classes = 10
num_experiences = 5
batch_size = 32
learning_rate = 0.001
momentum = 0.9

# Define normalization and transformation operations
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# Define evaluation plugin
eval_plugin = plugins.EvaluationPlugin(
    metrics.accuracy,
    save_best_model=True,
    compare_key='accuracy0'
)

# Define main function
def main():
    # Set seed value
    torch.manual_seed(42)

    # Create CLEAR benchmark
    benchmark = benchmarks.CLEARBenchmark(
        'datasets/CLEAR',
        num_experiences=num_experiences,
        train_transform=transform,
        eval_transform=transform,
        num_classes=num_classes
    )

    # Create ResNet18 model
    model = models.ResNet18(num_classes=num_classes)

    # Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define SGD optimizer and learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_experiences)

    # Define continual learning strategy
    strategy = training.Naive(
        model,
        optimizer,
        train_mb_size=batch_size,
        train_epochs=1,
        eval_mb_size=batch_size,
        device=device,
        evaluator=eval_plugin
    )

    # Set up logging
    logging.set_logging_level('INFO')
    logging.add_loggers([
        logging.TensorboardLogger(),
        logging.StdOutLogger(),
        logging.CSVLogger('results.csv')
    ])

    # Run training loop
    for experience in benchmark.train_stream:
        strategy.train(experience)
        strategy.eval(experience)

    # Generate accuracy matrix and compute CLEAR metrics
    accuracy_matrix = eval_plugin.accuracy_matrix
    clear_metrics = eval_plugin.compute_metrics()

    # Log results
    logging.info(f'Accuracy matrix: {accuracy_matrix}')
    logging.info(f'CLEAR metrics: {clear_metrics}')

# Run main function
if __name__ == '__main__':
    main()