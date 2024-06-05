```python
import argparse
from hydro import production_cost_function, add_and_assign
from mpisppy.utils import spin_the_wheel, write_spin_the_wheel_tree_solution, write_spin_the_wheel_first_stage_solution
from mpisppy.cylinders import Vanilla
from mpisppy.spbase import ScenarioCreator, ScenarioDenouement

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hydro MPISPPY model arguments")
    parser.add_argument("--branching-factors", nargs='+', type=int, help="Branching factors for the scenario tree")
    # Add more arguments as needed
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
    if len(args.branching_factors) < 1:
        raise ValueError("Branching factors must have at least one element.")
    
    node_names = [f"Node_{i}" for i in range(len(args.branching_factors))]
    scenario_names = [f"Scenario_{i}" for i in range(len(args.branching_factors))]
    
    scenario_creator = ScenarioCreator(production_cost_function)
    scenario_denouement = ScenarioDenouement(add_and_assign)
    
    # Setup for vanilla cylinders
    vanilla_cylinders = [Vanilla]
    
    # Configuration for spokes, if needed
    hub_dict = {}
    spoke_dict = {}
    
    # Example setup for Lagrangian bound and xhat looper bound spokes
    # This is placeholder logic; actual setup will depend on specific requirements
    if args.branching_factors[0] > 2:
        hub_dict["lagrangian_bound_spoke"] = {"option1": "value1"}
        spoke_dict["xhat_looper_bound_spoke"] = {"option2": "value2"}
    
    # Create and spin the wheel
    wheel = spin_the_wheel(scenario_creator, scenario_denouement, vanilla_cylinders, hub_dict, spoke_dict)
    
    # Print best inner and outer bounds
    print(f"Best inner bound: {wheel.best_inner_bound}")
    print(f"Best outer bound: {wheel.best_outer_bound}")
    
    # Condition to write solutions
    if wheel.best_inner_bound < 1000:
        write_spin_the_wheel_first_stage_solution(wheel)
        write_spin_the_wheel_tree_solution(wheel)

if __name__ == "__main__":
    main()
```