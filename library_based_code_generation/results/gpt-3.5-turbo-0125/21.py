```python
import rlberry
from rlberry.envs import bernoulli_bandit
from rlberry.agents.bandits import UCB, UCBV, ETC, MOSS, IMED, NPTS, EXP3
from rlberry.wrappers import WriterWrapper
from rlberry.experiment import ExperimentManager
import matplotlib.pyplot as plt
import seaborn as sns

env = bernoulli_bandit.BernoulliBandit(n_actions=5, success_prob=[0.1, 0.2, 0.3, 0.4, 0.5])

agents = [UCB, UCBV, ETC, MOSS, IMED, NPTS, EXP3]
agent_names = ['UCB', 'UCBV', 'ETC', 'MOSS', 'IMED', 'NPTS', 'EXP3']

for agent_class, agent_name in zip(agents, agent_names):
    agent = WriterWrapper(agent_class, env)
    agent.fit(env, horizon=1000)

    pseudo_regret = agent.pseudo_regret()
    regret = agent.regret()
    arm_selections = agent.arm_selections()

    plt.plot(pseudo_regret, label=agent_name)
    plt.plot(regret, label=agent_name)
    plt.plot(arm_selections, label=agent_name)

plt.xlabel('Time')
plt.ylabel('Cumulative Regret / Pseudo-Regret / Arm Selections')
plt.legend()
plt.show()
```