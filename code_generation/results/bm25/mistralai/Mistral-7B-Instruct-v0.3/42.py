 Here is a basic structure of the Python code using the Acme library. Please note that this is a simplified version and you might need to adjust it according to your specific environment and dataset.

```python
import numpy as np
import acme
import acme.learn as learn
import acme.buffers as buffers
import acme.agents.ddpg as ddpg
import acme.specs as specs
import openai_acme_env

class CRRAgent:
    def __init__(self, env_name, dataset_name, batch_size, evaluation_period, num_demonstration_episodes, random_seed, learning_rates, discount, target_update_period, use_sarsa_target):
        self.env = openai_acme_env.make_environment(env_name)
        self.dataset = acme.datasets.load(dataset_name)
        self.batch_size = batch_size
        self.evaluation_period = evaluation_period
        self.num_demonstration_episodes = num_demonstration_episodes
        self.random_seed = random_seed
        self.learning_rates = learning_rates
        self.discount = discount
        self.target_update_period = target_update_period
        self.use_sarsa_target = use_sarsa_target

    def add_next_action_extras(self, transitions):
        next_states, rewards, dones, actions, next_actions = zip(*transitions)
        next_states = np.stack(next_states)
        next_actions = np.stack(next_actions)
        transitions = list(zip(next_states, rewards, dones, actions, next_actions))
        return transitions

    def create_learner(self):
        policy = ddpg.policy.CategoricalPolicy(self.env.spec.observation_space, self.env.spec.action_space, self.learning_rates)
        qf1 = learn.QFunction(self.env.spec, self.learning_rates[0])
        qf2 = learn.QFunction(self.env.spec, self.learning_rates[1])
        optimizer = learn.Optimizer(qf1.target.parameters(), qf2.target.parameters())
        learner = ddpg.agent.DDPGAgent(policy, qf1, qf2, optimizer, self.discount, self.target_update_period, self.use_sarsa_target)
        return learner

    def create_evaluator(self):
        return ddpg.agent.Evaluator(self.env.spec, self.learning_rates[0])

    def run(self):
        np.random.seed(self.random_seed)
        self.dataset = self.dataset.replay(self.num_demonstration_episodes)
        transitions = self.dataset.transitions
        transitions = self.add_next_action_extras(transitions)
        buffer = buffers.ReplayBuffer(transitions, self.batch_size)
        learner = self.create_learner()
        evaluator = self.create_evaluator()
        evaluator_network = evaluator.network
        evaluator_network.eval()
        learner.set_evaluator(evaluator)
        loop = acme.learn.EnvironmentLoop(
            self.env,
            learner,
            buffer,
            max_path_length=self.env.spec.time_limit,
            discount=self.discount,
            evaluation_period=self.evaluation_period,
        )
        acme.learn.run_loop(loop)

if __name__ == "__main__":
    env_name = "MyEnvironment"
    dataset_name = "MyDataset"
    batch_size = 1000
    evaluation_period = 1000
    num_demonstration_episodes = 1000
    random_seed = 42
    learning_rates = [0.001, 0.0001]
    discount = 0.99
    target_update_period = 1000
    use_sarsa_target = False
    agent = CRRAgent(env_name, dataset_name, batch_size, evaluation_period, num_demonstration_episodes, random_seed, learning_rates, discount, target_update_period, use_sarsa_target)
    agent.run()
```

This code creates a CRR agent for a specified environment using the Acme library. The agent is configured with various parameters such as batch size, evaluation period, number of demonstration episodes, random seed, learning rates, discount, target update period, and whether to use SARSA target or not. The environment and dataset names are also configurable. The code includes a function to add next action extras to the transitions. The main function creates the environment, gets the demonstrations dataset, creates the networks to optimize, creates the learner, defines the evaluator network, creates the actor, and finally runs the environment loop.