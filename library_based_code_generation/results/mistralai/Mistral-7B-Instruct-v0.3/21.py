 Here is a simplified version of the Python code based on your instructions. Note that this code assumes you have the necessary libraries installed (rlberry, matplotlib, seaborn). Also, it's a basic example and might need adjustments to fit your specific needs.

```python
import numpy as np
import rlberry as rl
from rlberry.agents import QLearning, UCB, UCBV, ETC, MOSS, IMED, NPTS, EXP3
from rlberry.environments import BanditEnvironment
from rlberry.evaluation import CumulativeRegret, CumulativePseudoRegret, ActionCount
from rlberry.plotting import plot_cumulative_regret, plot_action_counts
from rlberry.tools.experiment_manager import ExperimentManager
from rlberry.tools.writer_wrapper import WriterWrapper

class BanditExperiment(ExperimentManager):
    def __init__(self, n_arms, n_trials, **kwargs):
        super().__init__(n_trials=n_trials, **kwargs)
        self.env = BanditEnvironment(n_arms=n_arms)
        self.agents = [
            QLearning(self.env),
            UCB(self.env),
            UCBV(self.env),
            ETC(self.env),
            MOSS(self.env),
            IMED(self.env),
            NPTS(self.env),
            EXP3(self.env)
        ]
        self.wrappers = [WriterWrapper(agent) for agent in self.agents]
        self.cumulative_pseudo_regret = CumulativePseudoRegret()
        self.cumulative_regret = CumulativeRegret()
        self.action_counts = ActionCount()

    def run(self):
        for trial in range(self.n_trials):
            for agent, wrapper in zip(self.agents, self.wrappers):
                for _ in range(self.n_episodes):
                    action = agent.act()
                    reward = self.env.reward(action)
                    wrapper.observe(action, reward)

        self.cumulative_pseudo_regret.update(self.wrappers)
        self.cumulative_regret.update(self.wrappers)
        self.action_counts.update(self.wrappers)

def main():
    n_arms = 10
    n_trials = 1000

    experiment = BanditExperiment(n_arms=n_arms, n_episodes=100, n_trials=n_trials)
    experiment.run()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plot_cumulative_regret(experiment.cumulative_pseudo_regret, ax=ax1)
    ax1.set_ylabel('Cumulative Pseudo-Regret')
    ax1.legend(loc='best')

    ax2 = ax1.twinx()
    plot_cumulative_regret(experiment.cumulative_regret, ax=ax2)
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig2, ax = plt.subplots(figsize=(10, 6))
    plot_action_counts(experiment.action_counts, ax=ax)
    ax.set_ylabel('Number of Times Each Arm Was Selected')
    ax.set_xlabel('Arms')
    ax.legend(loc='best')

    plt.show()

if __name__ == "__main__":
    main()
```

This code defines a `BanditExperiment` class that manages the experiment, runs it, and computes the cumulative pseudo-regret, cumulative regret, and action counts. The `main` function initializes the experiment, runs it, and plots the results using matplotlib and seaborn. The agents are wrapped with a `WriterWrapper` to track the action and reward.