 import acme
import tensorflow as tf
import sonnet as snt

flags = tf.flags

flags.DEFINE_string('environment_name', 'Pendulum-v0', 'Name of the environment.')
flags.DEFINE_integer('num_demonstrations', 100, 'Number of demonstrations to use.')
flags.DEFINE_integer('learning_steps', 100000, 'Number of learning steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate.')
flags.DEFINE_integer('hidden_units', 128, 'Number of hidden units.')
flags.DEFINE_integer('num_layers', 2, 'Number of layers.')
flags.DEFINE_boolean('is_distributed', False, 'Whether to run the experiment in a distributed manner.')

def create_demonstration_dataset_factory():
# Implement this function
pass

def create_environment_factory():
# Implement this function
pass

def create_network_factory():
def network_factory_fn(observation_spec, action_spec):
network = snt.nets.MLP(
output_sizes=[flags.hidden_units] * flags.num_layers + [action_spec.shape[-1]],
activation=tf.nn.relu,
dropout_rate=flags.dropout_rate)
return network
return network_factory_fn

def build_experiment_config():
experiment_logger_factory = create_experiment_logger_factory()
environment_factory = create_environment_factory()
demonstration_dataset_factory = create_demonstration_dataset_factory()
network_factory = create_network_factory()

if flags.is_distributed:
return make_distributed_experiment(
experiment_logger_factory,
environment_factory,
demonstration_dataset_factory,
network_factory,
learning_rate=flags.learning_rate,
batch_size=flags.batch_size)
else:
return make_experiment_config(
experiment_logger_factory,
environment_factory,
demonstration_dataset_factory,
network_factory,
learning_rate=flags.learning_rate,
batch_size=flags.batch_size)

def main():
experiment_config = build_experiment_config()
if flags.is_distributed:
run_experiment(make_distributed_offline_experiment(experiment_config))
else:
run_offline_experiment(experiment_config)

if __name__ == '__main__':
flags.mark_flag_as_required('environment_name')
tf.app.run()