 ```python
import acme
import acme.agents.crr
import acme.core
import acme.environment_loop
import acme.environments.suite
import acme.tf
import acme.tf.agents.crr
import acme.tf.learners
import acme.tf.networks
import tensorflow as tf
import numpy as np

def add_next_action_extras(transitions):
    next_actions = np.stack([t.next_action for t in transitions])
    extras = {
        'next_actions': tf.convert_to_tensor(next_actions),
    }
    for key, value in transitions[0].extras.items():
        extras[key] = tf.stack([t.extras[key] for t in transitions])
    for transition in transitions:
        transition.extras.update(extras)

def create_crr_agent(environment_name, dataset_name, batch_size, eval_period, num_demo_episodes, random_seed, lr_core, lr_head, discount, target_update_period, use_sarsa_target):
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    environment = acme.make_environment(environment_name)
    dataset = acme.dataset.Dataset.load(dataset_name, environment.observation_spec(), environment.action_spec())

    network = acme.tf.networks.CriticNetwork(
        environment.observation_spec(),
        environment.action_spec(),
        fc_layers=(256, 256),
        heads=(acme.tf.networks.ConstantHead(value=0.), acme.tf.networks.LearnedHead(fc_layers=(256, 256)))
    )

    learner = acme.tf.learners.Learner(
        network,
        acme.tf.optimizers.build_optimizer(lr_core),
        acme.tf.losses.ValueLoss(),
        tf.nest.map_structure(acme.tf.optimizers.build_optimizer(lr_head), network.heads),
        discount,
        target_update_period
    )

    evaluator_network = acme.tf.networks.CriticNetwork(
        environment.observation_spec(),
        environment.action_spec(),
        fc_layers=(256, 256),
        heads=(acme.tf.networks.ConstantHead(value=0.), acme.tf.networks.LearnedHead(fc_layers=(256, 256)))
    )

    actor = acme.tf.actors.DQNActor(network, evaluator_network, use_sarsa_target)

    agent = acme.tf.agents.crr.CRRAgent(
        environment,
        actor,
        learner,
        acme.tf.explorers.EpsilonGreedyExplorer(epsilon=0.1),
        tf.nest.map_structure(tf.constant, [batch_size]),
        add_next_action_extras,
        num_demo_episodes=num_demo_episodes,
        random_seed=random_seed
    )

    return agent

def main():
    agent = create_crr_agent(
        environment_name='acme/environment:random-v0',
        dataset_name='acme/dataset:random-v0',
        batch_size=64,
        eval_period=1000,
        num_demo_episodes=10,
        random_seed=42,
        lr_core=1e-3,
        lr_head=1e-3,
        discount=0.99,
        target_update_period=100,
        use_sarsa_target=False
    )

    environment_loop = acme.environment_loop.EnvironmentLoop(agent, acme.environment_loop.EpisodicLoop())
    environment_loop.collect(num_episodes=10000)

if __name__ == '__main__':
    main()
```