 import acme
import acme.agents.continuous as continuous_agent
import acme.core as core
import acme.environments.continuous as continuous_env
import acme.utils.loggers as loggers
import acme.utils.pickles as pickles
import launchpad as lp

# Define environment name and dataset name
environment_name = "Pendulum-v0"
demonstrations_dataset_name = "demonstrations_pendulum"

# Define various parameters
batch_size = 256
evaluation_period = 1000
num_demonstration_episodes = 10
random_seed = 0
learner_lr = 1e-3
critic_lr = 1e-3
cql_weight = 0.5
num_optimization_steps = 1

# Create environment
env = continuous_env.make(environment_name)

# Create dataset
demonstrations_dataset = pickles.load_dataset(demonstrations_dataset_name)

# Create agent
learner = continuous_agent.ContinuousQLearner(
env.observation_spec(),
env.action_spec(),
learner_lr,
cql_weight,
num_optimization_steps,
)
critic = continuous_agent.ContinuousCritic(
env.observation_spec(),
env.action_spec(),
critic_lr,
)
agent = continuous_agent.ContinuousQLearningAgent(learner, critic)

# Create evaluator network
evaluator = continuous_agent.ContinuousQLearningAgent(
learner,
continuous_agent.ContinuousCritic(
env.observation_spec(),
env.action_spec(),
critic_lr * 0.1,
),
)

# Create logger
logger = loggers.TerminalLogger(logger_name="cql_logger")

# Create environment loop
environment_loop = EnvironmentLoop(
env,
agent,
logger,
continuous_agent.CollectDemonstrations(demonstrations_dataset),
continuous_agent.CollectExperience(batch_size),
continuous_agent.Evaluate(evaluator, evaluation_period),
continuous_agent.StopAfter(num_demonstration_episodes),
)

# Run environment loop
environment_loop.run(num_episodes=10000, random_seed=random_seed)

# Save and restore agent
lp.launch(test_save_and_restore(agent))

# Update and wait
lp.launch(test_update_and_wait(agent))

# Apply and sample
lp.launch(test_apply_and_sample(agent))

# Normalize sample and update
lp.launch(test_normalize_sample_and_update(agent))

# Append and compute reward
lp.launch(test_append_and_compute_reward(agent))

# Compute discount and reward
lp.launch(test_compute_discount_and_reward(agent))