```python
import spiceypy as spice
from basilisk import architecture as bas
from basilisk.utilities import MonteCarloSimulation, SimulationExecutor

class Controller:
    def __init__(self):
        # spice.furnsh("path_to_spice_kernel")

    def load_spice_kernels(self):
        spice.furnsh("path_to_spice_kernel")

class MySimulation:
    def __init__(self, num_runs):
        self.num_runs = num_runs
        self.controller = Controller()
        self.simulation_executor = SimulationExecutor()

    def setup_simulation(self):
        # This function would ideally contain more setup logic
        pass

    def run_monte_carlo(self):
        for i in range(self.num_runs):
            self.setup_simulation()
            self.simulation_executor.execute_simulation()
            self.get_hubble_state()

    def get_hubble_state(self):
        # Assuming "HUBBLE" is the NAIF ID for the Hubble Space Telescope
        et = spice.str2et("2023-01-01")
        state, _ = spice.spkezr("HUBBLE", et, "J2000", "NONE", "EARTH")
        print(f"Hubble State at {et}: {state}")

def main():
    num_runs = 12
    my_simulation = MySimulation(num_runs)
    my_simulation.run_monte_carlo()
    # Clean up data after test
    spice.kclear()

if __name__ == "__main__":
    main()
```