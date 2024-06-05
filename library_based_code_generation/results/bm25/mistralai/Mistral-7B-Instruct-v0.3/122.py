 Here is a simplified Python code example using the FATE library for the specified binary classification task with a HeteroNN pipeline. Please note that this is a basic example and might need adjustments based on your specific dataset and requirements.

```python
import torch
import torch.nn as nn
import fate_flow as flow
import yaml

def load_config(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def create_models():
    guest_bottom = nn.Linear(config['input_dim'], config['hidden_dim'])
    guest_top = nn.Linear(config['hidden_dim'], 2)
    host_bottom = nn.Linear(config['input_dim'], config['hidden_dim'])
    interactive_layer = flow.InteractiveLayer(config['interactive_dim'])

    return guest_bottom, guest_top, host_bottom, interactive_layer

def create_heteronn(guest_bottom, guest_top, host_bottom, interactive_layer):
    learner = flow.HeteroNN()
    learner.add_bottom_model(guest_bottom, flow.Guest)
    learner.add_top_model(guest_top, flow.Guest)
    learner.add_bottom_model(host_bottom, flow.Host)
    learner.add_interactive_layer(interactive_layer)

    learner.set_optimizer(config['optimizer'])
    learner.set_loss_function(config['loss_function'])
    learner.set_task_type(flow.TaskType.CLASSIFICATION)
    learner.set_epochs(config['epochs'])
    learner.set_batch_size(config['batch_size'])
    learner.set_learning_rate(config['learning_rate'])

    return learner

def main(config_file):
    config = load_config(config_file)

    reader = flow.TextReader(config['data_path'])
    data_transform = flow.StatAndTransform(config['stat_and_transform'])
    intersection = flow.Intersection(data_transform)

    train_data = reader.read_data()
    guest_data, host_data = intersection.find_common_instances(train_data)

    data_transform.fit(guest_data)
    guest_data = data_transform.transform(guest_data)
    host_data = data_transform.transform(host_data)

    guest_bottom, guest_top, host_bottom, interactive_layer = create_models()
    learner = create_heteronn(guest_bottom, guest_top, host_bottom, interactive_layer)

    training_input_data = flow.TrainingInputData(guest_data, host_data)
    learner.fit(training_input_data)

    print(learner.summary())

if __name__ == "__main__":
    main("config.yaml")
```

This code assumes that you have a YAML configuration file named `config.yaml` with the following structure:

```yaml
input_dim: 10
hidden_dim: 20
interactive_dim: 5
optimizer: Adam
loss_function: CrossEntropyLoss
stat_and_transform:
  mean: [0.5, 0.5, ...]
  std: [0.5, 0.5, ...]
epochs: 10
batch_size: 32
learning_rate: 0.001
data_path: "path/to/your/data"
```