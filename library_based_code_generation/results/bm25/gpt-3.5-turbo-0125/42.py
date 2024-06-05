```python
import acme
from acme import environment_loop, specs, datasets, networks, actors, learners, evaluators

def add_next_action_extras(transitions):
    pass

def main():
    environment = make_environment('CartPole')
    dataset = get_demonstrations_dataset('CartPole')
    network = networks.DQN('CartPole')
    learner = learners.DQNLearner(network=network)
    evaluator_network = networks.DQN('CartPole')
    actor = actors.DQNAgent(network=network)
    
    loop = environment_loop.EnvironmentLoop(environment=environment,
                                            actor=actor,
                                            learner=learner,
                                            dataset=dataset,
                                            evaluator_network=evaluator_network,
                                            add_next_action_extras=add_next_action_extras,
                                            batch_size=32,
                                            evaluation_period=1000,
                                            num_demonstration_episodes=100,
                                            random_seed=42,
                                            learning_rate=1e-3,
                                            discount=0.99,
                                            target_update_period=1000,
                                            use_sarsa_target=True)
    
    loop.run()
```