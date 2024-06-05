 ```python
from Basilisk.utilities.MonteCarlo import Controller, RetentionPolicy, Dispersions, Scenario

def run(show_plots):
    controller = Controller()
    controller.set_simulation_function(scenario_AttFeedback.simulate)
    controller.set_execution_function(scenario_AttFeedback.execute)
    controller.set_execution_count(1000)
    controller.set_archive_directory('archive')
    controller.set_seed_dispersion('normal')
    controller.set_thread_count(4)
    controller.set_verbosity(2)
    controller.set_variable_casting(True)
    controller.set_dispersion_magnitude_file('magnitudes.txt')

    dispersions = [Dispersions.additive_gaussian, Dispersions.multiplicative_gaussian, Dispersions.jump]
    for dispersion in dispersions:
        controller.add_dispersion(dispersion)

    retention_policy = RetentionPolicy()
    retention_policy.add_message_log('info')
    retention_policy.set_data_callback(lambda data: displayPlots(data, retention_policy) if show_plots else None)
    controller.add_retention_policy(retention_policy)

    controller.execute_simulations()

def displayPlots(data, retention_policy):
    times, states = zip(*retention_policy.get_data())
    import matplotlib.pyplot as plt
    plt.plot(times, states)
    plt.xlabel('Time')
    plt.ylabel('States')
    plt.show()

if __name__ == "__main__":
    run(True)
```