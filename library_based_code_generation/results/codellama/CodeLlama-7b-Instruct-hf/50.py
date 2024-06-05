  import acme

def create_demonstration_dataset_factory(env_name, num_demos):
    return acme.DemonstrationDatasetFactory(env_name, num_demos)

def create_environment_factory(env_name):
    return acme.EnvironmentFactory(env_name)

def create_network_factory(num_layers, num_units, dropout_rate, learning_rate):
    return acme.NetworkFactory(num_layers, num_units, dropout_rate, learning_rate)

def build_experiment_configuration(env_name, num_demos, learning_steps, batch_size, learning_rate, dropout_rate, num_layers, num_units):
    demonstration_dataset_factory = create_demonstration_dataset_factory(env_name, num_demos)
    environment_factory = create_environment_factory(env_name)
    network_factory = create_network_factory(num_layers, num_units, dropout_rate, learning_rate)
    return acme.ExperimentConfiguration(demonstration_dataset_factory, environment_factory, network_factory, learning_steps, batch_size)

def main(env_name, num_demos, learning_steps, batch_size, learning_rate, dropout_rate, num_layers, num_units, distributed):
    experiment_configuration = build_experiment_configuration(env_name, num_demos, learning_steps, batch_size, learning_rate, dropout_rate, num_layers, num_units)
    if distributed:
        acme.run_experiment(experiment_configuration, distributed=True)
    else:
        acme.run_experiment(experiment_configuration, distributed=False)

if __name__ == "__main__":
    main()