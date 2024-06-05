 ```python
import argparse
import os
from typing import Dict, List

import hydro.hal as hal
import mpisppy.util.print_ as print_
import mpisppy.util.mpi_ as mpi
import mpisppy.cylinder_ as cylinder
import mpisppy.scenario_ as scenario
import mpisppy. wheel_ as wheel
import mpisppy.denouement_ as denouement

def create_config(args: argparse.Namespace) -> hal.Configuration:
    config = hal.Configuration()
    config.problem_name = args.problem_name
    config.num_scenarios = args.num_scenarios
    config.num_time_periods = args.num_time_periods
    config.branching_factors = [int(bf) for bf in args.branching_factors.split(',')]
    config.num_threads = args.num_threads
    config.verbose = args.verbose
    config.write_solutions = args.write_solutions
    config.first_stage_solution_file = args.first_stage_solution_file
    config.full_tree_solution_file = args.full_tree_solution_file
    return config

def main(args: argparse.Namespace):
    config = create_config(args)

    if len(config.branching_factors) != config.num_time_periods:
        print_("Number of branching factors must match number of time periods.")
        exit(1)

    node_names = [f"t{t+1}_b{b+1}" for t in range(config.num_time_periods) for b in range(config.branching_factors[t])]

    scenario_names = [f"scen_{s}" for s in range(config.num_scenarios)]

    scenario_creator = scenario.ScenarioCreator(
        config.problem_name,
        config.num_scenarios,
        config.num_time_periods,
        config.branching_factors,
        scenario_names,
        config.verbose,
    )

    denouement_creator = denouement.DenouementCreator(
        config.problem_name,
        config.num_scenarios,
        config.num_time_periods,
        config.branching_factors,
        scenario_names,
        config.verbose,
    )

    hub_dict = {
        "config": config,
        "scenario_creator": scenario_creator,
        "denouement_creator": denouement_creator,
    }

    spoke_dict = {
        "cylinder_creator": cylinder.CylinderCreator(
            config.problem_name,
            config.num_scenarios,
            config.num_time_periods,
            config.branching_factors,
            scenario_names,
            config.verbose,
        ),
    }

    if config.problem_name in ["lagrangian_bound", "xhat_looper_bound"]:
        spoke_dict["subproblem_creator"] = wheel.SubproblemCreator(
            config.problem_name,
            config.num_scenarios,
            config.num_time_periods,
            config.branching_factors,
            scenario_names,
            config.verbose,
        )

    wheel_spinner = wheel.WheelSpinner(
        config.problem_name,
        config.num_threads,
        node_names,
        hub_dict,
        spoke_dict,
        config.verbose,
    )

    wheel_spinner.spin()

    best_inner_bound = wheel_spinner.get_best_inner_bound()
    best_outer_bound = wheel_spinner.get_best_outer_bound()

    print_("Best inner bound:", best_inner_bound)
    print_("Best outer bound:", best_outer_bound)

    if config.write_solutions:
        if mpi.WORLD.rank == 0:
            wheel_spinner.write_solutions(
                config.first_stage_solution_file, config.full_tree_solution_file
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--problem_name", type=str, default="vanilla", help="Name of the problem")
    parser.add_argument("--num_scenarios", type=int, default=10, help="Number of scenarios")
    parser.add_argument("--num_time_periods", type=int, default=2, help="Number of time periods")
    parser.add_argument(
        "--branching_factors",
        type=str,
        default="2,2",
        help="Comma-separated list of branching factors for each time period",
    )
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--write_solutions", action="store_true", help="Write first stage and full tree solutions"
    )
    parser.add_argument(
        "--first_stage_solution_file",
        type=str,
        default="first_stage_solution.json",
        help="File name for the first stage solution",
    )
    parser.add_argument(
        "--full_tree_solution_file",
        type=str,
        default="full_tree_solution.json",
        help="File name for the full tree solution",
    )

    args = parser.parse_args()

    main(args)
```