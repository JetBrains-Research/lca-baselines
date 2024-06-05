 Here is the Python code based on your instructions, using the Basilisk library:

```python
import basilisk
import numpy as np

class Controller:
    def load_spice_kernels(self):
        # Load Spice kernels here
        pass

class MySimulation:
    def __init__(self):
        self.sim = basilisk.Simulation()
        self.controller = Controller()
        self.controller.load_spice_kernels()
        self.spacecraft = basilisk.Spacecraft()
        self.spacecraft.setInitialState(np.zeros(15))  # Initialize spacecraft state

    def get_hubble_state(self):
        hubble_state = self.sim.get_state(self.controller.get_spice_kernel(), 'HUBBLE')
        return hubble_state

    def simulation_function(self):
        hubble_state = self.get_hubble_state()
        print(hubble_state)

    def set_up_monte_carlo(self):
        self.sim.test_MonteCarloSimulation(self.simulation_function, 12)
        self.sim.setShouldArchiveParameters(True)
        self.sim.setShouldDisperseSeeds(True)

class Main:
    def __init__(self):
        self.sim = MySimulation()

    def run(self):
        self.sim.set_up_monte_carlo()
        self.sim.simulate()
        self.sim.cleanUp()

if __name__ == "__main__":
    main = Main()
    main.run()
```

This code sets up a simple Monte Carlo simulation using the Basilisk library. The `MySimulation` class initializes a spacecraft, loads the Spice kernels, and defines a simulation function that retrieves the Hubble state and prints it to the terminal. The `Main` class sets up the Monte Carlo simulation and runs it. The Spice kernel loading code in the `Controller` class is commented out, as per your instructions.