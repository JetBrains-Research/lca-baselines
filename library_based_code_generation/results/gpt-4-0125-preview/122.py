```python
import argparse
from fate_arch.common import log
from fate_flow.entity.metric import MetricMeta
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform, HeteroNN, Reader, Evaluation, Intersection
from pipeline.interface.model import Model
from pipeline.component.hetero_nn.backend.pytorch.models import DenseNN
from pipeline.component.hetero_nn.backend.pytorch.optim import SGD
from pipeline.component.hetero_nn.backend.pytorch.losses import BCELoss

def main(config="./config.yaml"):
    # Define the pipeline
    pipeline = PipeLine()

    # Reader component
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=10000).component_param(table={"name": "guest_table", "namespace": "experiment"})
    reader_0.get_party_instance(role='host', party_id=10001).component_param(table={"name": "host_table", "namespace": "experiment"})

    # DataTransform component
    data_transform_0 = DataTransform(name="data_transform_0")

    # Intersection component
    intersection_0 = Intersection(name="intersection_0")

    # HeteroNN component
    hetero_nn_0 = HeteroNN(name="hetero_nn_0", epochs=10, batch_size=64, learning_rate=0.01, task_type="binary")

    # Define models for HeteroNN
    guest_bottom_model = DenseNN(layers=[128, 64], input_shape=20, activation="relu")
    guest_top_model = DenseNN(layers=[64, 1], activation="sigmoid")
    host_bottom_model = DenseNN(layers=[128, 64], input_shape=30, activation="relu")
    interactive_layer = DenseNN(layers=[64, 32], activation="relu")

    # Add models to HeteroNN
    hetero_nn_0.add_bottom_model(guest_bottom_model, is_guest=True)
    hetero_nn_0.add_bottom_model(host_bottom_model, is_guest=False)
    hetero_nn_0.add_top_model(guest_top_model)
    hetero_nn_0.add_interactive_layer(interactive_layer)

    # Compile HeteroNN
    hetero_nn_0.compile(optimizer=SGD(lr=0.01), loss=BCELoss())

    # Evaluation component
    evaluation_0 = Evaluation(name="evaluation_0")

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(intersection_0, data=data_transform_0.output.data)
    pipeline.add_component(hetero_nn_0, data=intersection_0.output.data)
    pipeline.add_component(evaluation_0, data=hetero_nn_0.output.data)

    # Compile pipeline
    pipeline.compile()

    # Fit pipeline
    pipeline.fit()

    # Print summary
    print(pipeline.get_component("hetero_nn_0").get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a HeteroNN pipeline for binary classification.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.", default="./config.yaml")
    args = parser.parse_args()
    main(config=args.config)
```