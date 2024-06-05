 Here is a simplified example of how you might set up a pipeline using FATE (Fairness, Accountability, and Transparency) library in Python. Please note that this is a basic example and might need adjustments based on your specific use case. Also, FATE library is not officially released yet, so you might need to use the development version.

```python
import argparse
import yaml
from fate_flow.api.pipeline_service_client import PipelineServiceClient
from fate_flow.api.pipeline_service_pb2_grpc import PipelineServiceStub
from fate_flow.api.pipeline_service_pb2 import PipelineDef, DataReader, DataTransform, Intersection, FeatureScaler, FeatureBinning, DataStat, PearsonCorr, OneHotEncoder, FeatureSelector, LogisticRegression, Evaluator

def load_config(file):
    with open(file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def create_pipeline():
    config = load_config('config.yaml')

    # Define roles
    guest = PipelineDef.RoleType.GUEST
    host = PipelineDef.RoleType.HOST
    arbiter = PipelineDef.RoleType.ARBITER

    # Define steps
    steps = []

    # Data reading and transformation for guest and host
    data_reader_guest = DataReader(name='data_reader_guest')
    data_reader_host = DataReader(name='data_reader_host')
    steps.append(data_reader_guest)
    steps.append(data_reader_host)

    # Intersection
    intersection = Intersection(name='intersection')
    steps.append(intersection)

    # Feature scaling
    feature_scaler = FeatureScaler(name='feature_scaler')
    steps.append(feature_scaler)

    # Feature binning
    feature_binning = FeatureBinning(name='feature_binning')
    feature_binning_params = config['feature_binning_params']
    feature_binning.set_params(**feature_binning_params)
    steps.append(feature_binning)

    # Data statistics
    data_stat = DataStat(name='data_stat')
    steps.append(data_stat)

    # Pearson correlation
    pearson_corr = PearsonCorr(name='pearson_corr')
    steps.append(pearson_corr)

    # One-hot encoding
    one_hot_encoder = OneHotEncoder(name='one_hot_encoder')
    steps.append(one_hot_encoder)

    # Feature selection
    feature_selector = FeatureSelector(name='feature_selector')
    feature_selector_params = config['feature_selector_params']
    feature_selector.set_params(**feature_selector_params)
    steps.append(feature_selector)

    # Logistic regression
    logistic_regression = LogisticRegression(name='logistic_regression')
    logistic_regression_params = config['logistic_regression_params']
    logistic_regression.set_params(**logistic_regression_params)
    steps.append(logistic_regression)

    # Evaluation
    evaluator = Evaluator(name='evaluator')
    steps.append(evaluator)

    # Define pipeline
    pipeline = PipelineDef()
    pipeline.roles = [guest, host, arbiter]
    pipeline.steps = steps

    return pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    pipeline = create_pipeline()

    channel = grpc.insecure_channel('localhost:50051')
    stub = PipelineServiceStub(channel)
    response = stub.CreatePipeline(pipeline)

    print(f'Pipeline ID: {response.pipeline_id}')
```

This script creates a pipeline with the specified steps, defines roles for guest, host, and arbiter, and uses a configuration file to define parameters for feature binning, feature selection, and logistic regression. The pipeline is then compiled and fitted. If no configuration file is provided, it uses a default one named 'config.yaml'.