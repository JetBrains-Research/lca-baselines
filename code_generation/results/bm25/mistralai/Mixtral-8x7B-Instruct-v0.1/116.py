 ```python
import os
import torch
import torchvision
import torchvision.transforms as transforms
from avalanche.benchmarks import CelebA
from avalanche.models import ResNet18
from avalanche.training.strategies import Naive
from avalanche.training.learning_rates import StepDecay
from avalanche.output_writers import TensorboardWriter, CSVWriter
from avalanche.evaluation.metrics import AverageMeter, StopwatchMetric
from avalanche.logging import InteractiveLogger, ExperimentLogger

# Hyperparameters
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
GAMMA = 0.1
STEP_SIZE = 5
LOG_INTERVAL = 10
NUM_WORKERS = 2

def main():
 # Create a learning rate scheduler
 scheduler = StepDecay(LR, STEP_SIZE, GAMMA)

 # Initialize the model
 model = ResNet18(num_classes=2)

 # Define normalization and transformation operations for the training and testing data
 normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
 std=[0.5, 0.5, 0.5])

 train_transform = transforms.Compose([
 transforms.RandomHorizontalFlip(),
 transforms.RandomCrop(32, padding=4),
 transforms.ToTensor(),
 normalize,
 ])

 test_transform = transforms.Compose([
 transforms.ToTensor(),
 normalize,
 ])

 # Set up logging to Tensorboard, a text file, and stdout
 tb_logger = TensorboardWriter()
 csv_logger = CSVWriter()
 logger = InteractiveLogger(
 verbose=True,
 log_tensorboard=tb_logger,
 log_csv=csv_logger
 )

 # Define an evaluation plugin with various metrics
 metrics = [
 AverageMeter(),
 StopwatchMetric(),
 ]

 # Create a CLEAR benchmark
 benchmark = CelebA(
 'data/',
 phase='train',
 transform=train_transform,
 download=True,
 )

 # Set a seed value and create a CLEAR benchmark
 seed = 42
 test_benchmark = CelebA(
 'data/',
 phase='test',
 transform=test_transform,
 download=True,
 seed=seed,
 )
 test_benchmark.align_and_flatten_datasets()

 # Move the model to the appropriate device
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 model.to(device)

 # Define an SGD optimizer
 optimizer = torch.optim.SGD(
 model.parameters(),
 lr=LR,
 momentum=MOMENTUM,
 weight_decay=WEIGHT_DECAY
 )

 # Create a learning rate scheduler
 scheduler = StepDecay(LR, STEP_SIZE, GAMMA)

 # Create a continual learning strategy using the Naive method from Avalanche
 strategy = Naive(
 model=model,
 optimizer=optimizer,
 scheduler=scheduler,
 device=device,
 check_model_and_optimizer=check_model_and_optimizer,
 logger=logger,
 metrics=metrics,
 )

 # Run a training loop, saving the model after each experience and evaluating it on the test stream
 for experience in benchmark.train_stream:
 strategy.train(experience)
 strategy.eval(test_benchmark.get_dataset('test'))
 strategy.log_top_k_metrics(k=3)
 strategy.save_model(experience.name)

 # Generate an accuracy matrix and compute the CLEAR metrics, logging these results to a text file
 accuracy_matrix = strategy.get_accuracy_matrix()
 clear_metrics = strategy.compute_clear_metrics(accuracy_matrix)
 csv_logger.write_csv(clear_metrics, 'clear_metrics.csv')

if __name__ == '__main__':
 main()
```