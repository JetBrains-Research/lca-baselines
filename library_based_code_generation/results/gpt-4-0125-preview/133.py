```python
import argparse
from fate_arch.common import log
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform, HeteroNN, Evaluation, Intersection
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.runtime.entity import JobParameters
from pipeline.utils.tools import JobConfig

log.getLogger().setLevel(log.INFO)

def main(config="../../config.yaml"):
    # Load configuration
    if isinstance(config, str):
        config = JobConfig.load_from_file(config)
    
    # Data definition and pipeline initialization
    guest_train_data = {"name": config.guest_train_data["name"], "namespace": config.guest_train_data["namespace"]}
    host_train_data = {"name": config.host_train_data["name"], "namespace": config.host_train_data["namespace"]}
    
    guest_eval_data = {"name": config.guest_eval_data["name"], "namespace": config.guest_eval_data["namespace"]}
    host_eval_data = {"name": config.host_eval_data["name"], "namespace": config.host_eval_data["namespace"]}
    
    pipeline = PipeLine()

    # Reader Component
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=config.guest["party_id"]).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=config.host["party_id"]).component_param(table=host_train_data)

    # Data Transform Component
    data_transform_0 = DataTransform(name="data_transform_0")

    # Intersection Component
    intersection_0 = Intersection(name="intersection_0")

    # HeteroNN Component
    hetero_nn_0 = HeteroNN(name="hetero_nn_0", epochs=config.hetero_nn_params["epochs"],
                           batch_size=config.hetero_nn_params["batch_size"],
                           learning_rate=config.hetero_nn_params["learning_rate"],
                           early_stop=config.hetero_nn_params["early_stop"],
                           guest_bottom_model=config.hetero_nn_params["guest_bottom_model"],
                           host_bottom_model=config.hetero_nn_params["host_bottom_model"],
                           interactive_layer=config.hetero_nn_params["interactive_layer"],
                           guest_top_model=config.hetero_nn_params["guest_top_model"],
                           callback_param=config.hetero_nn_params["callback_param"])

    # Evaluation Component
    evaluation_0 = Evaluation(name="evaluation_0")

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_nn_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_nn_0.output.data))

    # Compile pipeline
    pipeline.compile()

    # Fit pipeline
    pipeline.fit(JobParameters(backend=0, work_mode=0))

    # Print HeteroNN Component Summary
    print(hetero_nn_0.get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HeteroNN pipeline")
    parser.add_argument("--config", type=str, help="Path to the configuration file", default="../../config.yaml")
    args = parser.parse_args()
    main(args.config)
```