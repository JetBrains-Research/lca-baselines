  import hydro
import mpisppy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branching_factors", nargs="+", type=int, required=True)
    parser.add_argument("--scenario_creator", type=str, required=True)
    parser.add_argument("--denouement", type=str, required=True)
    parser.add_argument("--spokes", nargs="+", type=int, required=False)
    parser.add_argument("--xhat_looper", action="store_true")
    parser.add_argument("--lagrangian_bound", action="store_true")
    parser.add_argument("--first_stage_solutions", action="store_true")
    parser.add_argument("--full_tree_solutions", action="store_true")
    return parser.parse_args()

def create_config(args):
    config = mpisppy.config.Configuration()
    config.branching_factors = args.branching_factors
    config.scenario_creator = args.scenario_creator
    config.denouement = args.denouement
    if args.spokes:
        config.spokes = args.spokes
    if args.xhat_looper:
        config.xhat_looper = True
    if args.lagrangian_bound:
        config.lagrangian_bound = True
    if args.first_stage_solutions:
        config.first_stage_solutions = True
    if args.full_tree_solutions:
        config.full_tree_solutions = True
    return config

def main():
    args = parse_args()
    config = create_config(args)
    scenario_names = [f"{i+1:02d}" for i in range(len(config.branching_factors))]
    node_names = [f"{i+1:02d}" for i in range(len(config.branching_factors))]
    if config.spokes:
        spoke_names = [f"{i+1:02d}" for i in range(len(config.spokes))]
    else:
        spoke_names = []
    hub = {node_name: scenario_name for node_name, scenario_name in zip(node_names, scenario_names)}
    spoke_dict = {spoke_name: hub for spoke_name in spoke_names}
    wheel_spinner = mpisppy.wheel_spinner.WheelSpinner(hub, spoke_dict)
    wheel_spinner.spin_wheel()
    best_inner_bound = wheel_spinner.best_inner_bound
    best_outer_bound = wheel_spinner.best_outer_bound
    if config.first_stage_solutions:
        first_stage_solutions = wheel_spinner.first_stage_solutions
    if config.full_tree_solutions:
        full_tree_solutions = wheel_spinner.full_tree_solutions
    print(f"Best inner bound: {best_inner_bound}")
    print(f"Best outer bound: {best_outer_bound}")
    if config.first_stage_solutions:
        print(f"First stage solutions: {first_stage_solutions}")
    if config.full_tree_solutions:
        print(f"Full tree solutions: {full_tree_solutions}")

if __name__ == "__main__":
    main()