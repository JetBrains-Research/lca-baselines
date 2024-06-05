```python
import acme
import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf

FLAGS = tf.flags.FLAGS

# Experiment configuration flags
tf.flags.DEFINE_string('environment_name', 'CartPole', 'Name of the environment')
tf.flags.DEFINE_integer('num_demonstrations', 100, 'Number of demonstrations')
tf.flags.DEFINE_integer('learning_steps', 10000, 'Number of learning steps')
tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
tf.flags.DEFINE_float('dropout_rate', 0.2, 'Dropout rate')
tf.flags.DEFINE_list('network_parameters', [128, 64], 'Network parameters for MLP')

def create_dataset_factory():
    pass

def create_environment_factory():
    pass

def create_network_factory():
    pass

def build_experiment_config():
    pass

def main():
    pass

if __name__ == '__main__':
    main()
```