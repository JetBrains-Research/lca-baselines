 import hydro
import mpisppy.utils.mpi as mpi
from mpisppy.scenariocreator import ScenarioCreator
from mpisppy.denouement import Denouement
from mpisppy.utils.argparse_utils import ArgumentParser
from mpisppy.utils.scenario_tree_utils import spin_the_wheel, write_spin_the_wheel_tree_solution, write_spin_the_wheel_first_stage_solution
from hydro.production_cost_function import production_cost_function

def create_configuration(args):
    configuration = hydro.Configuration()
    configuration.branching_factors = args.branching_factors
    configuration.scenario_names = [f"scenario_{i}" for i in range(1, sum(configuration.branching_factors)+1)]
    configuration.num_scenarios = sum(configuration.branching_factors) + 1
    configuration.num_time_periods = args.num_time_periods
    configuration.num_threads = args.num_threads
    configuration.num_processes = args.num_processes
    configuration.wheel_options = {"hub": {"name": "hub", "num_threads": configuration.num_threads},
                                  "spokes": [{"name": f"spoke_{i}", "num_threads": configuration.num_threads} for i in range(configuration.num_scenarios)]}
    return configuration

def main(args):
    args = parse_arguments(args)
    configuration = create_configuration(args)

    if max(configuration.branching_factors) > 1:
        node_names = [f"node_{i}" for i in range(1, sum(configuration.branching_factors)+1)]
    else:
        node_names = [f"node_{configuration.branching_factors[0]}"]

    scenario_creator = ScenarioCreator(configuration, node_names)
    denouement = Denouement(configuration)

    hub_args = {"name": "hub", "num_threads": configuration.num_threads, "num_processes": configuration.num_processes,
                "scenario_tree": scenario_creator.scenario_tree, "production_cost_function": production_cost_function,
                "scenario_creator": scenario_creator, "denouement": denouement}

    hub_dict = {"hub": hub_args}

    spoke_args_list = []
    for i in range(configuration.num_scenarios):
        spoke_args = {"name": f"spoke_{i}", "num_threads": configuration.num_threads, "num_processes": configuration.num_processes,
                      "scenario_tree": scenario_creator.scenario_tree, "production_cost_function": production_cost_function,
                      "scenario_creator": scenario_creator, "denouement": denouement}
        spoke_args_list.append(spoke_args)

    spokes_dict = {"spokes": spoke_args_list}

    wheel_dict = {"hub": hub_dict, "spokes": spokes_dict}

    spin_the_wheel(wheel_dict, configuration)

    if mpi.WORLD.rank == 0 and configuration.write_solutions:
        write_spin_the_wheel_tree_solution(scenario_creator.scenario_tree, denouement.solution, configuration)
        write_spin_the_wheel_first_stage_solution(scenario_creator.scenario_tree, denouement.solution, configuration)

if __name__ == "__main__":
    main(None)