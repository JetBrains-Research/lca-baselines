```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rlberry.envs.bandits import BernoulliBandit
from rlberry.manager import AgentManager, plot_writer_data
from rlberry.agents.bandits import UCB, UCBV, ETC, MOSS, IMED, NPTS, EXP3
from rlberry.wrappers import WriterWrapper

# Define the problem
n_arms = 5
arm_probabilities = np.random.rand(n_arms)
env = BernoulliBandit(n_arms=n_arms, arm_probabilities=arm_probabilities)

# Define agents
agent_list = [UCB, UCBV, ETC, MOSS, IMED, NPTS, EXP3]

# Parameters
horizon = 10000
n_episodes = 100

# ExperimentManager
managers = []
for agent in agent_list:
    agent_name = agent.__name__
    wrapped_agent = WriterWrapper(agent, writer_kwargs=dict(name=agent_name))
    manager = AgentManager(
        agent_class=wrapped_agent,
        env=env,
        fit_budget=horizon,
        n_fit=n_episodes,
        init_kwargs={'n_actions': n_arms},
        agent_name=agent_name
    )
    managers.append(manager)

# Train agents
for manager in managers:
    manager.fit()

# Plot cumulative pseudo-regret and cumulative regret
plt.figure(figsize=(12, 6))
sns.set()
for manager in managers:
    plot_writer_data(manager, tag='cumulative_regret', show=False, label=manager.agent_name, linestyle='--')
plt.title('Cumulative Regret')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for manager in managers:
    plot_writer_data(manager, tag='cumulative_pseudo_regret', show=False, label=manager.agent_name, linestyle='-')
plt.title('Cumulative Pseudo-Regret')
plt.legend()
plt.show()

# Plot the number of times each arm was selected
plt.figure(figsize=(12, 6))
for manager in managers:
    data = manager.agent.fit_info['n_pulls']
    plt.plot(data, label=manager.agent_name, linestyle=':')
plt.title('Number of times each arm was selected')
plt.xlabel('Arm')
plt.ylabel('Number of times selected')
plt.legend()
plt.show()
```