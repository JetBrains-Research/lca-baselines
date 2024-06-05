```python
# Set up and run a Continuous Q-Learning (CQL) agent on a specified environment using the Acme library

import acme
from acme import environment_loop
from acme import environments
from acme import datasets
from acme import networks
from acme.agents.tf import cql

# Define parameters
batch_size = 64
evaluation_period = 1000
num_demonstration_episodes = 100
random_seed = 42
learning_rate = 1e-3
cql_alpha = 0.2

# Create environment
environment_name = 'CartPole'
environment = environments.ContinuousEnvironment(environment_name)

# Obtain demonstrations dataset
dataset_name = 'demonstrations_dataset'
demonstrations_dataset = datasets.get_dataset(dataset_name)

# Create agent
agent = cql.CQL(
    environment_spec=environment.spec(),
    network=networks.DQN,
    batch_size=batch_size,
    evaluation_period=evaluation_period,
    num_demonstration_episodes=num_demonstration_episodes,
    random_seed=random_seed,
    learning_rate=learning_rate,
    cql_alpha=cql_alpha
)

# Create evaluator network
evaluator_network = networks.DQN

# Set up and run environment loop
loop = environment_loop.EnvironmentLoop(
    environment=environment,
    agent=agent,
    evaluator_network=evaluator_network
)
loop.run()
```