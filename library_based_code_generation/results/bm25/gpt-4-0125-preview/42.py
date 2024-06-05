```python
import acme
from acme import specs
from acme.agents.tf import crr
from acme.tf import networks
from acme.utils import loggers
import sonnet as snt
import tensorflow as tf
import dm_env
import numpy as np

def add_next_action_extras(transitions, next_actions):
    """Adds the next action to the extras of the transitions."""
    extras = transitions.extras
    if not extras:
        extras = {'next_actions': next_actions}
    else:
        extras.update({'next_actions': next_actions})
    return transitions._replace(extras=extras)

def main(environment_name, dataset_name, batch_size=256, evaluation_period=10,
         num_demonstration_episodes=100, seed=42, learning_rate=1e-3, discount=0.99,
         target_update_period=100, use_sarsa=False):
    # Create the environment
    environment = make_environment(environment_name)
    environment_spec = specs.make_environment_spec(environment)

    # Get the demonstrations dataset
    # Assuming the dataset is a list of episodes, each episode is a list of transitions.
    # This is a placeholder for loading a dataset. Replace with actual dataset loading.
    demonstrations = np.load(f"{dataset_name}.npy", allow_pickle=True)

    # Create the networks
    action_spec = environment_spec.actions
    num_actions = action_spec.num_values
    policy_network = snt.Sequential([
        networks.LayerNormMLP(layer_sizes=[256, 256, num_actions]),
        tf.nn.softmax
    ])
    critic_network = snt.Sequential([
        # Critic network takes both observations and actions as input
        lambda obs, act: tf.concat([obs, tf.one_hot(act, depth=num_actions)], axis=-1),
        networks.LayerNormMLP(layer_sizes=[256, 256, 1]),
        tf.squeeze
    ])

    # Create the agent (learner)
    agent = crr.CRR(
        environment_spec=environment_spec,
        policy_network=policy_network,
        critic_network=critic_network,
        batch_size=batch_size,
        observations_per_step=float(batch_size) / num_demonstration_episodes,
        target_policy_network=policy_network,
        target_critic_network=critic_network,
        min_replay_size=batch_size,
        max_replay_size=10000,
        discount=discount,
        target_update_period=target_update_period,
        dataset=demonstrations,
        learning_rate=learning_rate,
        use_sarsa=use_sarsa,
        seed=seed
    )

    # Define the evaluator
    evaluator_network = snt.Sequential([
        policy_network,
        lambda logits: tf.argmax(logits, axis=-1)
    ])

    # Create the actor
    actor = crr.CRRActor(policy_network=evaluator_network, adder=None)

    # Create the environment loop
    loop = acme.EnvironmentLoop(environment, actor, logger=loggers.TerminalLogger(label="Training", time_delta=0))

    # Run the environment loop
    for _ in range(evaluation_period):
        loop.run()

if __name__ == "__main__":
    main(environment_name="CartPole-v1", dataset_name="demonstrations")
```