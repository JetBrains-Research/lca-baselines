 Here is the Python code based on your instructions:

```python
import argparse
from mpi4py import MPI
from sppy.scenario import Scenario, ScenarioCreator
from sppy.extensions import Extension
from sppy.spokes import Spoke
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, nargs='+', required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--extensions', nargs='*', default=[])
    parser.add_argument('--spokes', nargs='*', default=[])
    parser.add_argument('--solution_dir', type=str)
    args = parser.parse_args()
    return args

def setup_config(config_file):
    # Assuming you have a function to load the configuration from a file
    # This is just a placeholder
    config = {}
    with open(config_file) as f:
        for line in f:
            key, value = line.strip().split(':')
            config[key] = value
    return config

def main(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    config = setup_config(args.config)
    creator = ScenarioCreator(config)

    available_scenarios = ['scenario1', 'scenario2', 'scenario3']
    if args.scenario not in available_scenarios:
        print(f"Error: Invalid scenario '{args.scenario}'")
        MPI.Abort(comm, 1)

    scenario = creator.create(args.scenario[0])

    extensions = []
    for ext in args.extensions:
        extensions.append(Extension(ext))

    spokes = []
    for spoke in args.spokes:
        spokes.append(Spoke(spoke))

    scenario.set_extensions(extensions)
    scenario.set_spokes(spokes)

    wheel_spinner = WheelSpinner(scenario, spokes)
    solution = wheel_spinner.spin()

    if args.solution_dir:
        solution_file = os.path.join(args.solution_dir, f"solution_{rank}.txt")
        with open(solution_file, 'w') as f:
            f.write(str(solution))

if __name__ == "__main__":
    args = parse_args()
    main(args)
```

Please note that this code assumes the existence of a `WheelSpinner` class, which is not part of the mpi4py or sppy libraries. You would need to define this class according to your specific requirements. Also, the function to load the configuration from a file is a placeholder and should be replaced with your actual configuration loading code.