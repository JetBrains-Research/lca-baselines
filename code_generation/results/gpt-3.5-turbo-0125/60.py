import basilisk

def rerun_monte_carlo(scenario_name, num_processes, run_numbers, retention_policies=[]):
    mc = basilisk.MonteCarloController()
    mc.set_initial_conditions_directory("initial_conditions")
    mc.set_archive_directory("archive")
    mc.set_execution_count(num_processes)
    mc.set_disperse_seeds(True)
    mc.set_archive_parameters(False)
    
    for policy in retention_policies:
        mc.add_retention_policy(policy)
    
    for run_number in run_numbers:
        mc.run(scenario_name, run_number)

if __name__ == "__main__":
    rerun_monte_carlo("scenario1", 4, [1, 2, 3], ["policy1", "policy2"])