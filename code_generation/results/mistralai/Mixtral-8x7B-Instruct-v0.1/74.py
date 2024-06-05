 ```python
import acme
import acme.agents.cql
import acme.builders.tf2
import acme.utils.loggers
import tensorflow as tf
import envs
import datasets

# Set up environment
environment_name = "my_environment"
environment = envs.make(environment_name)
dataset_name = "my_dataset"
demonstrations = datasets.load(dataset_name)

# Set up CQL agent
agent = acme.agents.cql.CQLAgent(
    environment=environment,
    network_factory=acme.builders.tf2.build_network,
    optimizer_factory=lambda: tf.keras.optimizers.Adam(learning_rate=1e-3),
    random_seed=0,
    batch_size=256,
    demonstrations=demonstrations,
    cql_temperature=0.1,
    cql_alpha=0.5,
    cql_beta=0.1,
    evaluation_interval=1000,
    num_evaluation_episodes=10,
)

# Set up evaluator network
evaluator_network = acme.builders.tf2.build_network(environment)
evaluator = acme.Agents.Evaluator(
    environment=environment,
    agent=agent,
    network=evaluator_network,
    num_episodes=10,
)

# Run agent in environment loop
logger = acme.utils.loggers.InMemoryLogger()
experiment = acme.EnvironmentLoop(
    environment=environment,
    agent=agent,
    logger=logger,
).collect(num_episodes=100000)

# Evaluate agent periodically
for i in range(100000, 101000, 1000):
    print(f"Evaluating at step {i}")
    metrics = evaluator.evaluate(num_episodes=10)
    print(metrics)
```