```python
import spiceypy as spice
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Assuming Basilisk setup and other necessary imports are handled elsewhere

class Controller:
    def __init__(self):
        # Load Spice kernels (this is commented out as per instruction, but typically would be uncommented for actual use)
        # spice.furnsh("path_to_kernel/meta_kernel.mk")

        pass

    def get_hubble_state(self):
        # This function would typically include calls to SPICE to get the Hubble state
        # For demonstration, we'll simulate this with a placeholder
        state = np.random.rand(6)  # Simulating a state vector
        return state

class MySimulation:
    def __init__(self, num_runs):
        self.num_runs = num_runs
        self.controller = Controller()

    def run_simulation(self, run_id):
        # Load Spice kernels for each thread (if needed)
        # spice.furnsh("path_to_kernel/meta_kernel.mk")

        # Simulate getting the Hubble state from SPICE for each run
        hubble_state = self.controller.get_hubble_state()
        print(f"Run {run_id}: Hubble State: {hubble_state}")

        # Here you would include the rest of the simulation steps

    def execute(self):
        with ThreadPoolExecutor(max_workers=self.num_runs) as executor:
            futures = [executor.submit(self.run_simulation, run_id) for run_id in range(self.num_runs)]
            for future in futures:
                future.result()  # Wait for all simulations to complete

def main():
    num_runs = 12
    simulation = MySimulation(num_runs)
    simulation.execute()

    # Cleanup or further data processing can go here

if __name__ == "__main__":
    main()
```