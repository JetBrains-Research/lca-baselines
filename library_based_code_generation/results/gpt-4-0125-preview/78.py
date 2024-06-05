```python
import argparse
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform, Intersection, HomoLR, Evaluation, Reader, FeatureBinning, OneHotEncoder, Sampler
from pipeline.component import SecureBoost
from pipeline.interface import Data

def main(config_path):
    # Load configuration
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Define the pipeline
    pipeline = PipeLine()

    # Reader components for guest and host
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=config['guest']['party_id']).component_param(table={"name": config['guest']['table_name'], "namespace": config['guest']['namespace']})
    reader_0.get_party_instance(role='host', party_id=config['host']['party_id']).component_param(table={"name": config['host']['table_name'], "namespace": config['host']['namespace']})

    # Data Transform
    data_transform_0 = DataTransform(name="data_transform_0")

    # Intersection
    intersection_0 = Intersection(name="intersection_0")

    # Sampler
    sampler_0 = Sampler(name="sampler_0")

    # Feature Binning
    feature_binning_0 = FeatureBinning(name="feature_binning_0")

    # OneHotEncoder
    one_hot_encoder_0 = OneHotEncoder(name="one_hot_encoder_0")

    # Logistic Regression
    homo_lr_0 = HomoLR(name="homo_lr_0")

    # SecureBoost
    secure_boost_0 = SecureBoost(name="secure_boost_0")

    # Evaluation
    evaluation_0 = Evaluation(name="evaluation_0")

    # Add components to the pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(sampler_0, data=Data(data=intersection_0.output.data))
    pipeline.add_component(feature_binning_0, data=Data(data=sampler_0.output.data))
    pipeline.add_component(one_hot_encoder_0, data=Data(data=feature_binning_0.output.data))
    pipeline.add_component(homo_lr_0, data=Data(train_data=one_hot_encoder_0.output.data))
    pipeline.add_component(secure_boost_0, data=Data(train_data=one_hot_encoder_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=[homo_lr_0.output.data, secure_boost_0.output.data]))

    # Compile and fit the pipeline
    pipeline.compile()
    pipeline.fit()

    # Print evaluation summary
    print(pipeline.get_component("evaluation_0").get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a federated learning task using FATE.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.config_path)
```