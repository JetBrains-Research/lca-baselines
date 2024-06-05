  from rlberry.manager import ExperimentManager
from rlberry.wrappers import WriterWrapper
from rlberry.agents import UCB, UCBV, ETC, MOSS, IMED, NPTS, EXP3
import matplotlib.pyplot as plt
import seaborn as sns

# Define problem parameters
n_arms = 10
n_rounds = 1000

# Define experiment
experiment = ExperimentManager(
    n_arms=n_arms,
    n_rounds=n_rounds,
    writer_wrapper=WriterWrapper()
)

# Define agents
agents = [
    UCB(experiment),
    UCBV(experiment),
    ETC(experiment),
    MOSS(experiment),
    IMED(experiment),
    NPTS(experiment),
    EXP3(experiment)
]

# Train agents
for agent in agents:
    agent.fit()

# Compute and plot cumulative pseudo-regret
cumulative_pseudo_regret = []
for agent in agents:
    cumulative_pseudo_regret.append(agent.writer_wrapper.cumulative_pseudo_regret)

plt.figure(figsize=(10, 6))
sns.lineplot(x=range(n_rounds), y=cumulative_pseudo_regret, label="Cumulative Pseudo-Regret")
plt.legend()
plt.show()

# Compute and plot cumulative regret
cumulative_regret = []
for agent in agents:
    cumulative_regret.append(agent.writer_wrapper.cumulative_regret)

plt.figure(figsize=(10, 6))
sns.lineplot(x=range(n_rounds), y=cumulative_regret, label="Cumulative Regret")
plt.legend()
plt.show()

# Compute and plot number of times each arm was selected
arm_selections = []
for agent in agents:
    arm_selections.append(agent.writer_wrapper.arm_selections)

plt.figure(figsize=(10, 6))
sns.lineplot(x=range(n_arms), y=arm_selections, label="Number of Times Each Arm Was Selected")
plt.legend()
plt.show()