```python
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics, confusion_matrix_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.models import SimpleMLP, ResNet18
from avalanche.training.strategies import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.generators import clear_benchmark

def main():
    # Hyperparameters
    lr = 0.01
    momentum = 0.9
    batch_size = 128
    epochs = 30
    log_every = 1
    seed = 1234

    # Normalization and transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Logging
    text_logger = TextLogger(open('log.txt', 'a'))
    tensorboard_logger = TensorboardLogger()
    interactive_logger = InteractiveLogger()

    # Evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=100, save_image=False, stream=True),
        loggers=[interactive_logger, text_logger, tensorboard_logger]
    )

    # Set the seed
    torch.manual_seed(seed)

    # Create the CLEAR benchmark
    benchmark = clear_benchmark(dataset_root='data', n_experiences=5, seed=seed)

    # Model
    model = ResNet18(num_classes=benchmark.n_classes)

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    # Learning rate scheduler
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    # Continual learning strategy
    cl_strategy = Naive(
        model, optimizer, torch.nn.CrossEntropyLoss(), lr_scheduler=lr_scheduler,
        train_mb_size=batch_size, train_epochs=epochs, eval_mb_size=batch_size,
        device=device, plugins=[], evaluator=eval_plugin, eval_every=log_every
    )

    # Training loop
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        cl_strategy.eval(benchmark.test_stream)

        # Save the model
        torch.save(model.state_dict(), f'model_after_exp_{experience.current_experience}.pth')

    # Generate accuracy matrix and compute CLEAR metrics
    # Note: Actual computation of CLEAR metrics would require additional implementation
    # based on the confusion matrix and other metrics collected during evaluation.
    # This part is left as an exercise to the reader due to the complexity and the
    # specific requirements of the CLEAR benchmark.

if __name__ == "__main__":
    main()
```