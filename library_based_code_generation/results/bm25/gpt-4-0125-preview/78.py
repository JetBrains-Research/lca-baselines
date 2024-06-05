```python
import argparse
from fate_arch.common import log
from fate_flow.entity import MetricMeta
from fate_flow.entity.metric import Metric, MetricData
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import Evaluation
from pipeline.component import FeatureBinning
from pipeline.component import HeteroFeatureBinning
from pipeline.component import HeteroLR
from pipeline.component import HeteroSecureBoost
from pipeline.component import Intersection
from pipeline.component import LocalBaseline
from pipeline.component import OneHotEncoder
from pipeline.component import Reader
from pipeline.component import Sampler
from pipeline.interface import Data
from pipeline.runtime.entity import JobParameters
from pipeline.runtime.entity import Role
from pipeline.utils.tools import load_job_config

log.getLogger().setLevel(log.logging.INFO)


def main(config="../../config.yaml"):
    # Load configuration
    job_config = load_job_config(config)

    # Define roles
    guest = job_config["role"]["guest"][0]
    host = job_config["role"]["host"][0]

    # Create a pipeline
    pipeline = PipeLine()

    # Set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    pipeline.set_roles(guest=guest, host=host)

    # Reader components
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=job_config["data"]["guest"])
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=job_config["data"]["host"])

    # Data transform components
    data_transform_0 = DataTransform(name="data_transform_0")

    # Sampler component
    sampler_0 = Sampler(name="sampler_0")

    # Feature binning component
    feature_binning_0 = HeteroFeatureBinning(name="feature_binning_0")

    # One-hot encoder component
    one_hot_encoder_0 = OneHotEncoder(name="one_hot_encoder_0")

    # Intersection component
    intersection_0 = Intersection(name="intersection_0")

    # Logistic Regression component
    hetero_lr_0 = HeteroLR(name="hetero_lr_0")

    # Local Baseline component
    local_baseline_0 = LocalBaseline(name="local_baseline_0")

    # Secure Boosting component
    hetero_secureboost_0 = HeteroSecureBoost(name="hetero_secureboost_0")

    # Evaluation component
    evaluation_0 = Evaluation(name="evaluation_0")

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(sampler_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(feature_binning_0, data=Data(data=sampler_0.output.data))
    pipeline.add_component(one_hot_encoder_0, data=Data(data=feature_binning_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=one_hot_encoder_0.output.data))
    pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(local_baseline_0, data=Data(data=intersection_0.output.data))
    pipeline.add_component(hetero_secureboost_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=[hetero_lr_0.output.data, hetero_secureboost_0.output.data]))

    # Compile pipeline
    pipeline.compile()

    # Fit pipeline
    pipeline.fit(job_parameters=JobParameters(work_mode=job_config["work_mode"], job_type=job_config["job_type"]))

    # Print evaluation summary
    print(pipeline.get_component("evaluation_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a federated learning pipeline using FATE.")
    parser.add_argument("--config", type=str, help="Path to the configuration file", default="../../config.yaml")
    args = parser.parse_args()
    main(config=args.config)
```