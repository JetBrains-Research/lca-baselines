 Here is the Python code that implements the given instructions using the avalanche library:

```python
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from avalanche.utils.transforms import Normalize
from avalanche.datasets import MNIST
from avalanche.models import SimpleMLP
from avalanche.metrics import Accuracy
from avalanche.model_selection import create_multi_dataset_generic_scenario
from avalanche.memory import ReservoirSamplingBuffer
from avalanche.strategies.replay import Replay
from avalanche.utils.common import _to_device
from avalanche.utils.progress import TqdmUpTo
from avalanche.utils.misc import freeze_up_to
from avalanche.utils.plotting import plot_metrics
from avalanche.utils.logger import get_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    return parser.parse_args()

def setup_model(input_dim, hidden_dim, output_dim):
    model = SimpleMLP(input_dim, hidden_dim, output_dim)
    return model

def setup_scenario(args, dataset, batch_size, num_workers):
    scenario = create_multi_dataset_generic_scenario(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_after_n_examples=1000,
        shuffle_each_iteration=True,
        task_reset_method='forget',
        task_reset_after_n_examples=10000,
        task_reset_after_n_iterations=100,
        task_reset_after_n_epochs=1,
        task_reset_after_n_batches=1,
        task_reset_after_n_sequential_tasks=1,
        task_reset_after_n_consecutive_failures=1,
        task_reset_after_n_consecutive_successes=1,
        task_reset_after_n_examples_since_last_reset=10000,
        task_reset_after_n_iterations_since_last_reset=100,
        task_reset_after_n_epochs_since_last_reset=1,
        task_reset_after_n_batches_since_last_reset=1,
        task_reset_after_n_sequential_tasks_since_last_reset=1,
        task_reset_after_n_consecutive_failures_since_last_reset=1,
        task_reset_after_n_consecutive_successes_since_last_reset=1,
        task_reset_after_n_examples_since_last_success=10000,
        task_reset_after_n_iterations_since_last_success=100,
        task_reset_after_n_epochs_since_last_success=1,
        task_reset_after_n_batches_since_last_success=1,
        task_reset_after_n_sequential_tasks_since_last_success=1,
        task_reset_after_n_consecutive_failures_since_last_success=1,
        task_reset_after_n_consecutive_successes_since_last_success=1,
        task_reset_after_n_examples_since_last_failure=10000,
        task_reset_after_n_iterations_since_last_failure=100,
        task_reset_after_n_epochs_since_last_failure=1,
        task_reset_after_n_batches_since_last_failure=1,
        task_reset_after_n_sequential_tasks_since_last_failure=1,
        task_reset_after_n_consecutive_failures_since_last_failure=1,
        task_reset_after_n_consecutive_successes_since_last_failure=1,
        task_reset_after_n_examples_since_last_reset_per_task=10000,
        task_reset_after_n_iterations_since_last_reset_per_task=100,
        task_reset_after_n_epochs_since_last_reset_per_task=1,
        task_reset_after_n_batches_since_last_reset_per_task=1,
        task_reset_after_n_sequential_tasks_since_last_reset_per_task=1,
        task_reset_after_n_consecutive_failures_since_last_reset_per_task=1,
        task_reset_after_n_consecutive_successes_since_last_reset_per_task=1,
        task_reset_after_n_examples_since_last_success_per_task=10000,
        task_reset_after_n_iterations_since_last_success_per_task=100,
        task_reset_after_n_epochs_since_last_success_per_task=1,
        task_reset_after_n_batches_since_last_success_per_task=1,
        task_reset_after_n_sequential_tasks_since_last_success_per_task=1,
        task_reset_after_n_consecutive_failures_since_last_success_per_task=1,
        task_reset_after_n_consecutive_successes_since_last_success_per_task=1,
        task_reset_after_n_examples_since_last_failure_per_task=10000,
        task_reset_after_n_iterations_since_last_failure_per_task=100,
        task_reset_after_n_epochs_since_last_failure_per_task=1,
        task_reset_after_n_batches_since_last_failure_per_task=1,
        task_reset_after_n_sequential_tasks_since_last_failure_per_task=1,
        task_reset_after_n_consecutive_failures_since_last_failure_per_task=1,
        task_reset_after_n_consecutive_successes_since_last_failure_per_task=1,
        task_reset_after_n_examples_since_last_reset_per_task_per_class=10000,
        task_reset_after_n_iterations_since_last_reset_per_task_per_class=100,
        task_reset_after_n_epochs_since_last_reset_per_task_per_class=1,
        task_reset_after_n_batches_since_last_reset_per_task_per_class=1,
        task_reset_after_n_sequential_tasks_since_last_reset_per_task_per_class=1,
        task_reset_after_n_consecutive_failures_since_last_reset_per_task_per_class=1,
        task_reset_after_n_consecutive_successes_since_last_reset_per_task_per_class=1,
        task_reset_after_n_examples_since_last_success_per_task_per_class=10000,
        task_reset_after_n_iterations_since_last_success_per_task_per_class=100,
        task_reset_after_n_epochs_since_last_success_per_task_per_class=1,
        task_reset_after_n_batches_since_last_success_per_task_per_class=1,
        task_reset_after_n_sequential_tasks_since_last_success_per_task_per_class=1,
        task_reset_after_n_consecutive_failures_since_last_success_per_task_per_class=1,
        task_reset_after_n_consecutive_successes_since_last_success_per_task_per_class=1,
        task_reset_after_n_examples_since_last_failure_per_task_per_class=10