```python
import acme
from acme import specs
from acme.agents.tf import bc
from acme.tf import networks
from acme.utils import loggers
import sonnet as snt
import tensorflow as tf
import gym

# Define flags for experiment configuration
ENV_NAME = 'CartPole-v1'
NUM_DEMONSTRATIONS = 1000
LEARNING_STEPS = 10000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.5
NETWORK_LAYERS = [64, 64]

def create_environment_factory(env_name):
    def environment_factory():
        environment = gym.make(env_name)
        environment = acme.wrappers.GymWrapper(environment)
        return environment
    return environment_factory

def create_demonstration_dataset_factory(env_factory, num_demonstrations):
    # This is a placeholder function. In practice, you would load or generate your demonstrations here.
    def demonstration_dataset_factory():
        # Generate or load demonstrations
        # This should return a dataset of 'num_demonstrations' demonstrations from the environment
        pass
    return demonstration_dataset_factory

def create_network_factory(action_spec, layers, dropout_rate):
    def network_factory():
        return snt.Sequential([
            snt.nets.MLP(layers, activation=tf.nn.relu, dropout_rate=dropout_rate),
            networks.NearZeroInitializedLinear(output_size=action_spec.num_values),
        ])
    return network_factory

def build_experiment_config(env_name, num_demonstrations, learning_steps, batch_size, learning_rate, dropout_rate, network_layers):
    env_factory = create_environment_factory(env_name)
    demonstration_dataset_factory = create_demonstration_dataset_factory(env_factory, num_demonstrations)
    environment = env_factory()
    action_spec = specs.make_continuous_action_spec(environment.action_spec().minimum, environment.action_spec().maximum)

    network_factory = create_network_factory(action_spec, network_layers, dropout_rate)

    bc_builder = bc.BCBuilder(learning_rate=learning_rate, batch_size=batch_size, dataset_factory=demonstration_dataset_factory)

    return acme.ExperimentConfig(
        environment_factory=env_factory,
        network_factory=network_factory,
        builder=bc_builder,
        policy_network=network_factory,
        max_num_actor_steps=learning_steps,
        log_every_step=1000,
    )

def main():
    experiment_config = build_experiment_config(
        ENV_NAME,
        NUM_DEMONSTRATIONS,
        LEARNING_STEPS,
        BATCH_SIZE,
        LEARNING_RATE,
        DROPOUT_RATE,
        NETWORK_LAYERS
    )

    # Choose between distributed or single-threaded execution based on your setup
    # For simplicity, we'll use a single-threaded executor here
    local_executor = acme.LocalExecutor(experiment_config, loggers.make_default_logger('experiment', time_delta=10))
    local_executor.run()

if __name__ == '__main__':
    main()
```