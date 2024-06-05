```python
import rlberry.seeding as seeding
from rlberry.envs import GridWorld
from rlberry.agents import UCB, UCBV, ETC, MOSS, IMED, NPTS, EXP3
from rlberry.wrappers import WriterWrapper
from rlberry.experiment import ExperimentManager
import matplotlib.pyplot as plt
import seaborn as sns

# Define parameters of the problem
env = GridWorld(nrows=10, ncols=10)
horizon = 1000

# Construct the experiment
seeding.set_global_seed(1234)
experiment = ExperimentManager(env, horizon)

# Define classes of agents
agents = [
    UCB(env.action_space.n),
    UCBV(env.action_space.n),
    ETC(env.action_space.n),
    MOSS(env.action_space.n),
    IMED(env.action_space.n),
    NPTS(env.action_space.n),
    EXP3(env.action_space.n)
]

# Wrap agents with WriterWrapper
for agent in agents:
    agent = WriterWrapper(agent)

# Train agents
experiment.train_agents(agents)

# Compute and plot cumulative pseudo-regret and cumulative regret
experiment.plot_regret()

# Compute and plot number of times each arm was selected
experiment.plot_action_counts()

plt.show()
```