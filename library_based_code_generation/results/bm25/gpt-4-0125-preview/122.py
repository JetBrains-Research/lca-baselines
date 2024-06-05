```python
import argparse
import torch
import torch.nn as nn
from fate_arch.common import log
from fate_flow.entity.metric import Metric, MetricMeta
from fate_flow.components.component_base import FateFlowComponent
from fate_flow.components.modules import HeteroNN
from fate_flow.components.modules import Reader, DataTransform, Intersection, Evaluation
from fate_flow.components.pipeline import Pipeline

class GuestBottomModel(nn.Module):
    def __init__(self):
        super(GuestBottomModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class GuestTopModel(nn.Module):
    def __init__(self):
        super(GuestTopModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class HostBottomModel(nn.Module):
    def __init__(self):
        super(HostBottomModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x

def main(config_path):
    # Load configuration
    config = {}
    with open(config_path, 'r') as f:
        config = eval(f.read())

    # Define the pipeline
    pipeline = Pipeline()

    # Add Reader component
    pipeline.add_component(Reader(name="reader"))

    # Add DataTransform component
    pipeline.add_component(DataTransform(name="data_transform"))

    # Add Intersection component
    pipeline.add_component(Intersection(name="intersection"))

    # Define HeteroNN component
    hetero_nn = HeteroNN(name="hetero_nn", epochs=config['epochs'], batch_size=config['batch_size'], learning_rate=config['learning_rate'], task_type="classification")
    hetero_nn.add_bottom_model(GuestBottomModel(), is_guest=True)
    hetero_nn.add_bottom_model(HostBottomModel(), is_guest=False)
    hetero_nn.add_top_model(GuestTopModel())
    hetero_nn.set_interactive_layer(nn.Linear(128, 128))
    hetero_nn.compile(optimizer="adam", loss="binary_crossentropy")

    # Add HeteroNN component to the pipeline
    pipeline.add_component(hetero_nn, data=DataTransform.name)

    # Add Evaluation component
    pipeline.add_component(Evaluation(name="evaluation"), data=HeteroNN.name)

    # Compile and fit the pipeline
    pipeline.compile()
    pipeline.fit(config)

    # Print the summary of HeteroNN component
    hetero_nn.summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a HeteroNN pipeline for binary classification.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.config)
```