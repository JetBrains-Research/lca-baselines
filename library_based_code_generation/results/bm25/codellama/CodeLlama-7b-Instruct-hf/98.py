  import hydro
import mpisppy

def parse_args():
    # Define arguments and create configuration object
    args = mpisppy.utils.arg_parsing.parse_args()
    config = mpisppy.config.Configuration(args)
    return args, config

def main():
    # Parse arguments and create configuration object
    args, config = parse_args()

    # Check length of branching factors and create node names
    if len(config.branching_factors) != 2:
        raise ValueError("Branching factors must be length 2")
    node_names = [f"{i+1}x{j+1}" for i in range(config.branching_factors[0]) for j in range(config.branching_factors[1])]

    # Create scenario names and set up scenario creator and denouement
    scenario_names = [f"{node_name}_scenario" for node_name in node_names]
    scenario_creator = mpisppy.scenario.ScenarioCreator(config, scenario_names)
    denouement = mpisppy.denouement.Denouement(config, scenario_creator)

    # Prepare necessary arguments for 'vanilla' cylinders
    cylinder_args = {"node_names": node_names, "scenario_names": scenario_names}

    # Set up spokes for Lagrangian bound and xhat looper bound
    if config.use_lagrangian_bound:
        lagrangian_spoke = mpisppy.spoke.LagrangianSpoke(config, scenario_creator, denouement)
    if config.use_xhat_looper_bound:
        xhat_looper_spoke = mpisppy.spoke.XhatLooperSpoke(config, scenario_creator, denouement)

    # Create wheel spinner with hub and spoke dictionaries
    hub = {"node_names": node_names, "scenario_names": scenario_names}
    spoke_dict = {"lagrangian": lagrangian_spoke, "xhat_looper": xhat_looper_spoke}
    wheel_spinner = mpisppy.wheel_spinner.WheelSpinner(hub, spoke_dict)

    # Spin the wheel and print best inner and outer bounds
    best_inner_bound, best_outer_bound = wheel_spinner.spin_the_wheel()
    print(f"Best inner bound: {best_inner_bound}")
    print(f"Best outer bound: {best_outer_bound}")

    # Write first stage and full tree solutions if necessary
    if config.write_solutions:
        wheel_spinner.write_spin_the_wheel_tree_solution()
        wheel_spinner.write_spin_the_wheel_first_stage_solution()

if __name__ == "__main__":
    main()