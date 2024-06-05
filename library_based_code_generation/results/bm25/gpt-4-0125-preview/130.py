```python
import argparse
from mpi4py import MPI
import mpisppy.utils.sputils as sputils
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import config
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Set up configuration for scenario-based optimization")
    parser.add_argument("--num_scenarios", type=int, help="Number of scenarios")
    parser.add_argument("--scenario_creator", type=str, help="Path to the scenario creator function")
    parser.add_argument("--solution_directory", type=str, default=None, help="Directory to write solution files")
    args = parser.parse_args()
    cfg = config.Config()
    cfg.num_scenarios = args.num_scenarios
    cfg.scenario_creator = args.scenario_creator
    cfg.solution_directory = args.solution_directory
    return cfg

def main():
    cfg = parse_args()
    
    if cfg.num_scenarios not in [10, 20, 30, 40, 50]:
        print(f"Number of scenarios {cfg.num_scenarios} is not supported.")
        return
    
    scenario_creator = sputils.import_module(cfg.scenario_creator).scenario_creator
    scenario_names = [f"Scenario{i}" for i in range(1, cfg.num_scenarios + 1)]
    
    # Example setup for extensions and spokes, adjust as needed
    extensions = None
    if cfg.num_scenarios == 10:
        extensions = ["ext1", "ext2"]  # Placeholder for actual extension setup
    spokes = None
    if cfg.num_scenarios == 20:
        spokes = ["spoke1", "spoke2"]  # Placeholder for actual spoke setup
    
    # Create the WheelSpinner object
    wheel = WheelSpinner(scenario_creator, scenario_names, extensions=extensions, spokes=spokes)
    
    # Spin the wheel
    wheel.spin()
    
    # Write solution to file if solution directory is provided
    if cfg.solution_directory:
        if not os.path.exists(cfg.solution_directory):
            os.makedirs(cfg.solution_directory)
        solution_file = os.path.join(cfg.solution_directory, "solution.txt")
        with open(solution_file, "w") as f:
            f.write("Solution details here")  # Placeholder for actual solution writing logic

if __name__ == "__main__":
    main()
```