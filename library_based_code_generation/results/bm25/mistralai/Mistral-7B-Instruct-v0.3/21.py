 Here is a simplified version of the Python code using the rlberry library, matplotlib, and seaborn. Please note that this is a basic example and might need adjustments based on your specific requirements.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rlberry import Agents, Environments, ExperimentManager, WriterWrapper

# Define the bandit problem
num_arms = 10
num_trials = 1000

env = Environments.Bandit(num_arms=num_arms, rewards=np.random.normal(loc=0, scale=1, size=(num_arms, num_trials)))

# Define the agents
UCB = Agents.UCB1(num_arms=num_arms)
UCBV = Agents.UCB2(num_arms=num_arms)
ETC = Agents.EpsilonGreedy(num_arms=num_arms, epsilon=0.1)
MOSS = Agents.MOSS(num_arms=num_arms)
IMED = Agents.IMED(num_arms=num_arms)
NPTS = Agents.NPTS(num_arms=num_arms)
EXP3 = Agents.EXP3(num_arms=num_arms, c=0.1)

# Wrap the agents with WriterWrapper
agents = [WriterWrapper(agent) for agent in [UCB, UCBV, ETC, MOSS, IMED, NPTS, EXP3]]

# Create and fit the experiment manager
experiment_manager = ExperimentManager(env, agents, num_trials)
experiment_manager.fit()

# Compute and plot the cumulative pseudo-regret and cumulative regret for each agent
pseudo_regret, regret = experiment_manager.compute_regret()
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.lineplot(data=pseudo_regret, dashes=[(4, 2) for _ in range(len(agents))], label='Pseudo-Regret', ax=ax1)
ax1.set_ylabel('Cumulative Regret')
ax1.legend(loc='lower right')

ax2 = ax1.twinx()
sns.lineplot(data=regret, color='r', marker='o', markersizes=(5,), label='Regret', ax=ax2)
ax2.set_ylabel('Cumulative Regret')
ax2.legend(loc='upper right')
plt.show()

# Compute and plot the number of times each arm was selected
arm_selections = experiment_manager.get_arm_selections()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=arm_selections.index, y=arm_selections.values, ax=ax)
plt.show()
```

This code defines a bandit problem with 10 arms, trains several bandit agents using the rlberry library, and plots the cumulative pseudo-regret, cumulative regret, and the number of times each arm was selected. The agents are wrapped with WriterWrapper to track the action and reward. The experiment is managed using the ExperimentManager class. The plots are created using matplotlib and seaborn, with varying line styles for each agent.