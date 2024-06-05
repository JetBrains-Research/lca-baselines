```python
import acme
from acme import specs
from acme.agents.tf import cql
from acme.tf import networks
from acme.utils import loggers
import sonnet as snt
import tensorflow as tf
import dm_env
import numpy as np
import gym

# Environment setup
environment_name = 'CartPole-v1'
environment = gym.make(environment_name)
environment = dm_env.specs.make_environment_spec(environment)

# Parameters
batch_size = 256
evaluation_period = 10
num_demonstration_episodes = 20
random_seed = 42
learning_rate = 1e-3
cql_alpha = 1.0
cql_tau = 10.0

# Fix the random seed for reproducibility
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# Network setup
observation_spec, action_spec, _ = environment
actor_network = networks.LayerNormMLP((256, 256, action_spec.num_values))
critic_network = networks.CriticMultiplexer(
    critic_network=networks.LayerNormMLP((256, 256, 1)),
    action_network=networks.ClipToSpec(action_spec)
)

# Optimizer
optimizer = snt.optimizers.Adam(learning_rate)

# Dataset setup (assuming dataset is available in the specified format)
dataset_name = 'your_dataset_name_here'
# Load or generate your demonstrations dataset here
# This is a placeholder for dataset loading which you will need to replace with actual data loading code
demonstrations_dataset = None  # Replace with code to load your dataset

# CQL Agent
agent = cql.CQL(
    environment_spec=environment,
    policy_network=actor_network,
    critic_network=critic_network,
    observation_network=tf.identity,  # Use the identity if observations are not pre-processed
    dataset=demonstrations_dataset,
    batch_size=batch_size,
    samples_per_insert=2,
    learning_rate=learning_rate,
    discount=0.99,
    target_update_period=100,
    cql_alpha=cql_alpha,
    cql_tau=cql_tau,
    seed=random_seed
)

# Loop setup
loop = acme.EnvironmentLoop(
    environment,
    agent,
    logger=loggers.TerminalLogger(time_delta=evaluation_period),
)

# Run the loop
loop.run(num_episodes=num_demonstration_episodes)
```