 ```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rlberry
from rlberry.agents.bandits import UCB, UCBV, ETC, MOSS, IMED, NPTS, EXP3
from rlberry.envs.bandits import BernoulliBandit
from rlberry.experiment import Experiment, WriterWrapper
from rlberry.experiment.policy_class import EpsilonGreedy

# Experiment parameters
T = 10000
num_arms = 10
num_agents = 7

# Environment
env = BernoulliBandit(num_arms=num_arms)

# Agents
agents = [
    WriterWrapper(UCB(num_arms=num_arms), "UCB"),
    WriterWrapper(UCBV(num_arms=num_arms), "UCBV"),
    WriterWrapper(ETC(num_arms=num_arms), "ETC"),
    WriterWrapper(MOSS(num_arms=num_arms), "MOSS"),
    WriterWrapper(IMOSS(num_arms=num_arms), "IMOSS"),
    WriterWrapper(NPTS(num_arms=num_arms), "NPTS"),
    WriterWrapper(EXP3(num_arms=num_arms), "EXP3"),
]

# Experiment manager
manager = ExperimentManager(
    env,
    agents,
    T,
    EpsilonGreedy(num_actions=num_arms, epsilon=0.1),
    n_parallel=1,
    max_repetitions=1,
)
manager.run()

# Compute and plot cumulative pseudo-regret
cumulative_pseudo_regret = np.array(
    [agent.cumulative_pseudo_regret for agent in agents]
)
plt.plot(cumulative_pseudo_regret, label="Cumulative pseudo-regret")
plt.xlabel("Time steps")
plt.ylabel("Cumulative pseudo-regret")
plt.legend()
plt.show()

# Compute and plot cumulative regret
cumulative_regret = np.array([agent.cumulative_regret for agent in agents])
plt.plot(cumulative_regret, label="Cumulative regret")
plt.xlabel("Time steps")
plt.ylabel("Cumulative regret")
plt.legend()
plt.show()

# Plot the number of times each arm was selected
arm_selection_counts = np.array(
    [agent.action_counts for agent in agents]
).T
plt.plot(arm_selection_counts, linewidth=2)
plt.xlabel("Arms")
plt.ylabel("Number of selections")
plt.legend(agents)
plt.show()
```