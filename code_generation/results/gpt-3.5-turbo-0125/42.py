import acme
import dm_env
import numpy as np

def add_next_action_extras(transitions):
    # Function to add next action extras to transitions
    return transitions

def main():
    # Create environment
    environment = dm_env.Environment()

    # Get demonstrations dataset
    dataset = acme.datasets.Demonstrations()

    # Create networks to optimize
    network = acme.networks.DQN()

    # Create the learner
    learner = acme.agents.CRR(network=network)

    # Define the evaluator network
    evaluator_network = acme.networks.DQN()

    # Create the actor
    actor = acme.agents.Actor()

    # Create the environment loop
    environment_loop = acme.environment_loop.EnvironmentLoop()

    # Run the environment loop
    environment_loop.run()

if __name__ == '__main__':
    main()