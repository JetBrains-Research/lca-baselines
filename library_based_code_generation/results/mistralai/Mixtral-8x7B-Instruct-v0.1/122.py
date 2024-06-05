 ```python
import torch
import fate
from torch import optim
from torch.nn import BCEWithLogitsLoss

def create_guest_bottom_model():
    # Define and return the guest bottom model
    pass

def create_guest_top_model():
    # Define and return the guest top model
    pass

def create_host_bottom_model():
    # Define and return the host bottom model
    pass

def create_interactive_layer():
    # Define and return the interactive layer
    pass

def main(config_file):
    reader = fate.Reader(config_file)
    data_transform = fate.DataTransform(config_file)
    intersection = fate.Intersection(config_file)

    guest_bottom_model = create_guest_bottom_model()
    guest_top_model = create_guest_top_model()
    host_bottom_model = create_host_bottom_model()
    interactive_layer = create_interactive_layer()

    hetero_nn = fate.HeteroNN(guest_bottom_model, guest_top_model, host_bottom_model, interactive_layer,
                              task_type='binary_classification', epochs=10, learning_rate=0.01, batch_size=32)

    optimizer = optim.Adam(hetero_nn.parameters())
    loss_fn = BCEWithLogitsLoss()
    hetero_nn.compile(optimizer, loss_fn)

    train_data = reader.read(data_transform, 'train')
    hetero_nn.fit(train_data, intersection)

    print(hetero_nn.summary())
```
Please note that the actual implementation of the models and layers is not provided in this code snippet. You will need to implement these functions according to your specific use case and data requirements.