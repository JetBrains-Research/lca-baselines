  from Basilisk.utilities.MonteCarlo import Controller, RetentionPolicy, Dispersions
from scenario_AttFeedback import scenario
import matplotlib.pyplot as plt

def run(show_plots):
    # Create a Monte Carlo simulation controller
    controller = Controller()

    # Set the simulation and execution functions
    controller.simulation_function = scenario.simulate
    controller.execution_function = scenario.execute

    # Set the execution count
    controller.execution_count = 1000

    # Set the archive directory
    controller.archive_directory = "./results"

    # Set the seed dispersion
    controller.seed_dispersion = Dispersions.Uniform(0, 1000)

    # Set the thread count
    controller.thread_count = 4

    # Set the verbosity
    controller.verbosity = 1

    # Set the variable casting
    controller.variable_casting = {"x": float, "y": float}

    # Set the dispersion magnitude file
    controller.dispersion_magnitude_file = "./dispersion_magnitude.txt"

    # Define a list of dispersions
    dispersions = [
        Dispersions.Uniform(0, 10),
        Dispersions.Normal(5, 2),
        Dispersions.LogNormal(1, 0.5)
    ]

    # Add the dispersions to the Monte Carlo controller
    for dispersion in dispersions:
        controller.add_dispersion(dispersion)

    # Create a retention policy
    retention_policy = RetentionPolicy()

    # Add message logs to the retention policy
    retention_policy.add_message_log("simulation_time")
    retention_policy.add_message_log("simulation_state")

    # Set the data callback
    retention_policy.data_callback = displayPlots

    # Add the retention policy to the Monte Carlo controller
    controller.add_retention_policy(retention_policy)

    # Execute the simulations
    controller.execute_simulations()

    # Execute callbacks if 'show_plots' is True
    if show_plots:
        retention_policy.execute_callbacks()

    # Return
    return

def displayPlots(data, retention_policy):
    # Extract time and states from the data
    time = data["simulation_time"]
    states = data["simulation_state"]

    # Plot the states against time
    plt.plot(time, states)
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.show()

    # Return
    return

if __name__ == "__main__":
    run(True)