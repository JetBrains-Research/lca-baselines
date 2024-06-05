```python
import acme
from acme import specs
from acme.agents.tf import cql
from acme.tf import networks
from acme.tf import utils

# Set up environment
environment_name = 'CartPole'
environment = gym.make(environment_name)
environment_spec = specs.make_environment_spec(environment)

# Set up demonstrations dataset
dataset_name = 'demonstrations_dataset'
demonstrations_dataset = load_dataset(dataset_name)

# Set up CQL agent
batch_size = 64
evaluation_period = 1000
num_demonstration_episodes = 100
random_seed = 42
learning_rate = 1e-3
cql_alpha = 0.2
cql_temperature = 1.0

network = networks.DQN(environment_spec.actions.num_values)
network_optimizer = tf.keras.optimizers.Adam(learning_rate)
target_network = networks.DQN(environment_spec.actions.num_values)
target_network_optimizer = tf.keras.optimizers.Adam(learning_rate)

agent = cql.CQL(
    environment_spec=environment_spec,
    network=network,
    target_network=target_network,
    dataset=demonstrations_dataset,
    batch_size=batch_size,
    num_demonstration_episodes=num_demonstration_episodes,
    cql_alpha=cql_alpha,
    cql_temperature=cql_temperature,
    random_seed=random_seed,
    network_optimizer=network_optimizer,
    target_network_optimizer=target_network_optimizer
)

# Set up evaluator network
evaluator_network = networks.DQN(environment_spec.actions.num_values)

# Run agent in environment loop
loop = acme.EnvironmentLoop(environment, agent)
loop.run(num_episodes=10000)
```