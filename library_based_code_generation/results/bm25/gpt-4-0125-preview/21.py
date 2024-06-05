```python
import matplotlib.pyplot as plt
import seaborn as sns
from rlberry.agents import UCBVAgent, ETCBanditAgent, MOSSAgent, IMEDAgent, NPTSAgent, Exp3Agent
from rlberry.envs.bandit import BernoulliBandit
from rlberry.manager import AgentManager, ExperimentManager
from rlberry.wrappers import WriterWrapper

# Define the bandit problem
n_arms = 10
arm_probabilities = [0.05 * (i+1) for i in range(n_arms)]
env = BernoulliBandit(arm_probabilities)

# Define agents
agent_classes = [UCBVAgent, ETCBanditAgent, MOSSAgent, IMEDAgent, NPTSAgent, Exp3Agent]
agent_params = {'horizon': 10000}

# Wrap agents with WriterWrapper
wrapped_agent_classes = {cls.__name__: WriterWrapper(cls) for cls in agent_classes}

# Create and configure ExperimentManager
experiment_manager = ExperimentManager()

for agent_name, agent_class in wrapped_agent_classes.items():
    manager = AgentManager(agent_class, env, agent_params, init_kwargs={'n_actions': n_arms}, n_fit=10, output_dir=f'./{agent_name}_output')
    experiment_manager.append(manager)

# Run the experiment
experiment_manager.run()

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Cumulative pseudo-regret
for manager in experiment_manager:
    data = manager.get_writer_data()
    sns.lineplot(x='episode', y='cumulative_pseudo_regret', data=data, ax=axs[0], label=manager.agent_name)

axs[0].set_title('Cumulative Pseudo-Regret')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Pseudo-Regret')

# Cumulative regret
for manager in experiment_manager:
    data = manager.get_writer_data()
    sns.lineplot(x='episode', y='cumulative_regret', data=data, ax=axs[1], label=manager.agent_name)

axs[1].set_title('Cumulative Regret')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Regret')

plt.legend()
plt.tight_layout()
plt.show()

# Plot the number of times each arm was selected
fig, ax = plt.subplots(figsize=(12, 6))
for manager in experiment_manager:
    data = manager.get_writer_data()
    for arm in range(n_arms):
        arm_data = data[data['action'] == arm]
        sns.lineplot(x='episode', y='action_count', data=arm_data, ax=ax, label=f'{manager.agent_name} - Arm {arm}', linestyle='--')

ax.set_title('Number of Times Each Arm Was Selected')
ax.set_xlabel('Episode')
ax.set_ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()
```