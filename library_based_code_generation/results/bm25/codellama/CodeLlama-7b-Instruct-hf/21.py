  ```
import rlberry
from rlberry.agents import UCB, UCBV, ETC, MOSS, IMED, NPTS, EXP3
from rlberry.wrappers import WriterWrapper
from rlberry.manager import ExperimentManager
import matplotlib.pyplot as plt
import seaborn as sns

# Define the problem parameters
num_arms = 10
num_rounds = 1000

# Define the experiment
experiment = ExperimentManager(
    num_arms=num_arms,
    num_rounds=num_rounds,
    env_name="Bandit",
    agent_classes=[UCB, UCBV, ETC, MOSS, IMED, NPTS, EXP3],
    agent_kwargs={"num_arms": num_arms},
    env_kwargs={"num_arms": num_arms},
)

# Train the agents
experiment.train()

# Compute and plot the cumulative pseudo-regret and cumulative regret for each agent
cumulative_pseudo_regret = experiment.get_cumulative_pseudo_regret()
cumulative_regret = experiment.get_cumulative_regret()
plt.plot(cumulative_pseudo_regret, label="Cumulative Pseudo-Regret")
plt.plot(cumulative_regret, label="Cumulative Regret")
plt.legend()
plt.show()

# Compute and plot the number of times each arm was selected
num_selections = experiment.get_num_selections()
plt.plot(num_selections, label="Number of Selections")
plt.legend()
plt.show()
```
This code defines the problem parameters, constructs the experiment using the `ExperimentManager` class, trains the agents using the `train()` method, computes and plots the cumulative pseudo-regret and cumulative regret for each agent, and computes and plots the number of times each arm was selected. The agents are wrapped with a `WriterWrapper` to track the action and reward. The experiment is managed using the `ExperimentManager` class, which provides methods for training and evaluating the agents. The plots are created using `matplotlib` and `seaborn`, with varying line styles for each agent.