 import argparse
import acme
import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import tree

def network_factory(input_shape, num_actions):
    def network(observations):
        with snt.as_layer():
            x = snt.Relu(name='relu_0')(snt.Linear(name='linear_0')(observations))
            x = snt.Dropout(rate=dropout_rate, name='dropout_0')(x)
            x = snt.Relu(name='relu_1')(snt.Linear(name='linear_1')(x))
            x = snt.Dropout(rate=dropout_rate, name='dropout_1')(x)
            return snt.Linear(num_actions, name='output')(x)
    return network

def dataset_factory(environment_factory, num_episodes):
    def dataset():
        environment = environment_factory()
        dataset = []
        for _ in range(num_episodes):
            timestep = environment.reset()
            while not timestep.last():
                action = timestep.observation['action']
                next_timestep = environment.step(action)
                dataset.append((timestep.observation, action, next_timestep.reward, next_timestep.observation))
                timestep = next_timestep
        return tf.data.Dataset.from_tensor_slices(tree.map_structure(lambda x: np.array(x, dtype=np.float32), dataset))
    return dataset

def environment_factory(environment_name):
    return lambda: acme.make_environment(environment_name)

def experiment_config(
        environment_factory,
        dataset_factory,
        network_factory,
        num_learning_steps,
        batch_size,
        learning_rate,
        dropout_rate,
        network_params):
    environment = environment_factory()
    dataset = dataset_factory(environment_factory, num_demonstrations)
    network = network_factory(environment.observation_spec(), environment.action_spec().num_values)
    learner = acme.tf2.Learner(
        network=network,
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        batch_size=batch_size,
        replay_buffer=acme.ReplayBuffer(capacity=100_000),
        random_seed=0)
    algorithm = acme.tf2.BehavioralCloning(
        learner=learner,
        dataset=dataset,
        num_steps=num_learning_steps,
        random_seed=0)
    return acme.EnvironmentLoop(environment, algorithm).run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment-name', type=str, default='CartPole-v1')
    parser.add_argument('--num-demonstrations', type=int, default=1000)
    parser.add_argument('--num-learning-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--dropout-rate', type=float, default=0.1)
    parser.add_argument('--network-params', type=str, default='{"sizes": [64, 64]}')
    args = parser.parse_args()

    num_demonstrations = args.num_demonstrations
    num_learning_steps = args.num_learning_steps
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout_rate = args.dropout_rate
    network_params_str = args.network_params
    network_params = tree.map_structure(lambda x: int(x), eval(network_params_str))

    experiment_config(
        environment_factory,
        dataset_factory,
        network_factory,
        num_learning_steps,
        batch_size,
        learning_rate,
        dropout_rate,
        network_params)