```python
import acme
from acme import specs
from acme.agents.tf import actors
from acme.tf import networks
from acme.tf import utils as tf_utils
from acme.utils import loggers
import sonnet as snt
import tensorflow as tf
import tensorflow_datasets as tfds

# Define flags for configuring the experiment
flags.DEFINE_string('environment_name', 'CartPole-v1', 'Name of the environment.')
flags.DEFINE_integer('num_demonstrations', 100, 'Number of demonstrations.')
flags.DEFINE_integer('learning_steps', 10000, 'Number of learning steps.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for the optimizer.')
flags.DEFINE_float('dropout_rate', 0.5, 'Dropout rate for the network.')
flags.DEFINE_list('network_layers', [64, 64], 'Sizes of the network layers.')

FLAGS = flags.FLAGS

def create_environment_factory(environment_name):
    def environment_factory():
        return gym.make(environment_name)
    return environment_factory

def create_demonstration_dataset_factory(num_demonstrations):
    def demonstration_dataset_factory():
        # This is a placeholder function. In practice, you would load your demonstrations here.
        # For example, you might use tensorflow_datasets to load a dataset.
        return tfds.load('your_dataset_name', split=f'train[:{num_demonstrations}]')
    return demonstration_dataset_factory

def create_network_factory(input_spec, output_spec):
    def network_factory():
        return snt.Sequential([
            snt.nets.MLP(list(map(int, FLAGS.network_layers)), activation=tf.nn.relu, dropout_rate=FLAGS.dropout_rate),
            networks.PolicyValueHead(output_spec.num_values)
        ])
    return network_factory

def build_experiment_config():
    environment_factory = create_environment_factory(FLAGS.environment_name)
    demonstration_dataset_factory = create_demonstration_dataset_factory(FLAGS.num_demonstrations)
    agent_network_factory = create_network_factory

    return acme.ExperimentConfig(
        environment_factory=environment_factory,
        agent_builder=acme.agents.tf.bc.BehavioralCloningBuilder(
            learning_rate=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size,
            dataset_factory=demonstration_dataset_factory,
            network_factory=agent_network_factory
        ),
        network_factory=agent_network_factory,
        num_episodes=None,
        max_episode_length=None,
    )

def main():
    # Parse command line flags.
    app.run(lambda argv: None)

    # Build the experiment configuration.
    experiment_config = build_experiment_config()

    # Run the experiment.
    if FLAGS.distributed:
        make_distributed_offline_experiment(experiment_config)
    else:
        run_experiment(experiment_config)

if __name__ == '__main__':
    main()
```