```python
import acme
from acme import specs
from acme.agents.tf import crr
from acme.tf import networks
from acme.utils import loggers
from acme.datasets import tfds
import sonnet as snt
import tensorflow as tf
import numpy as np

def add_next_action_extras(transitions, next_actions):
    return transitions._replace(extras={'next_actions': next_actions})

def main(environment_name, dataset_name, batch_size=256, evaluation_period=100, num_demonstration_episodes=10, random_seed=42, learning_rate=1e-3, discount=0.99, target_update_period=100, use_sarsa=False):
    # Set the random seed for reproducibility
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    # Create the environment
    environment = acme.make_environment_spec(environment_name)
    environment_spec = specs.make_environment_spec(environment)

    # Get the demonstrations dataset
    demonstrations_dataset = tfds.get_tfds_dataset(dataset_name, num_demonstration_episodes)

    # Create the networks to optimize
    action_spec = environment_spec.actions
    num_actions = action_spec.num_values
    policy_network = snt.Sequential([
        networks.LayerNormMLP(layer_sizes=[256, 256, num_actions]),
        tf.nn.softmax
    ])
    critic_network = snt.Sequential([
        networks.CriticMultiplexer(
            critic_network=networks.LayerNormMLP(layer_sizes=[256, 256, 1]),
            action_network=networks.ClipToSpec(action_spec)
        )
    ])

    # Create the learner
    learner = crr.CRR(
        environment_spec=environment_spec,
        policy_network=policy_network,
        critic_network=critic_network,
        dataset=demonstrations_dataset,
        discount=discount,
        target_update_period=target_update_period,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_sarsa=use_sarsa
    )

    # Define the evaluator network
    evaluator_network = snt.Sequential([
        policy_network,
        lambda logits: tf.argmax(logits, axis=-1)
    ])

    # Create the actor
    actor = crr.CRRActor(policy_network=evaluator_network, adder=None)

    # Create the environment loop
    loop = acme.EnvironmentLoop(environment, actor, logger=loggers.TerminalLogger())

    # Run the environment loop
    for _ in range(evaluation_period):
        loop.run_episode()

if __name__ == "__main__":
    main(environment_name='CartPole-v1', dataset_name='cartpole')
```