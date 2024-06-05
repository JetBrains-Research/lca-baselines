 Here is a basic structure of the Python script using FATE library as per your instructions. Please note that this is a simplified version and you may need to adjust it according to your specific needs and the exact API usage in your FATE environment.

```python
import argparse
import yaml
from fate_flow.api.chainer_compiler import ChainerCompiler
from fate_flow.api.chainer_client import ChainerClient
from fate_flow.api.chainer_server import ChainerServer
from fate_flow.api.data_and_variable import init_data_and_variable
from fate_flow.api.data_and_header_alignment import data_and_header_alignment
from fate_flow.api.hetero_feature_binning_guest_runner import hetero_feature_binning_guest_runner
from fate_flow.api.hetero_feature_binning_host_runner import hetero_feature_binning_host_runner
from fate_flow.api.hetero_feature_selection_runner import hetero_feature_selection_runner
from fate_flow.api.hetero_feature_selection_param import HeteroFeatureSelectionParam
from fate_flow.api.feature_binning_param import FeatureBinningParam
from fate_flow.api.feature_binning_converter import FeatureBinningConverter
from fate_flow.api.homo_feature_binning import HomoFeatureBinning
from fate_flow.api.hetero_feature_binning import HeteroFeatureBinning
from fate_flow.api.base_feature_binning import BaseFeatureBinning
from fate_flow.api.hetero_feature_binning_guest import HeteroFeatureBinningGuest
from fate_flow.api.hetero_feature_binning_host import HeteroFeatureBinningHost
from fate_flow.api.quantile_binning_and_count import quantile_binning_and_count
from fate_flow.api.set_feature import set_feature
from fate_flow.api.classification_and_regression_extract import _classification_and_regression_extract
from fate_flow.api.feature_binning_converter import FeatureBinningConverter
from fate_flow.api.hetero_feature_binning import HeteroFeatureBinning
from chainer import serializers

def load_config(file_path):
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def main(config):
    # Initialize data and variable
    data, variable = init_data_and_variable(config['data']['path'])

    # Data and header alignment
    data, variable = data_and_header_alignment(data, variable)

    # Guest roles: data reading, data transformation, intersection, and feature scaling
    guest_pipeline = ChainerClient()
    guest_pipeline.run_module(hetero_feature_binning_guest_runner, args=[data, variable, config['guest']['feature_binning']])
    guest_pipeline.run_module(HeteroFeatureBinningGuest, args=[data, variable])
    guest_pipeline.run_module(HeteroFeatureBinning, args=[data, variable])
    guest_pipeline.run_module(set_feature, args=['label'])

    # Host roles: feature scaling, feature binning, feature statistics, Pearson correlation, one-hot encoding, feature selection
    host_pipeline = ChainerServer()
    host_pipeline.run_module(HeteroFeatureBinningHost, args=[data, variable])
    host_pipeline.run_module(BaseFeatureBinning, args=[data, variable, FeatureBinningParam()])
    host_pipeline.run_module(HeteroFeatureBinning, args=[data, variable])
    host_pipeline.run_module(quantile_binning_and_count, args=[data, variable])
    host_pipeline.run_module(PearsonCorrelation, args=[data, variable])
    host_pipeline.run_module(OneHotEncoder, args=[data, variable])
    host_pipeline.run_module(HeteroFeatureSelection, args=[data, variable, HeteroFeatureSelectionParam()])

    # Compile and fit the pipeline
    compiler = ChainerCompiler()
    model = compiler.compile(host_pipeline)
    model.fit(data, epochs=config['training']['epochs'])

    # Evaluate the model
    predictions = model.predict(data)
    evaluation_result = _evaluate_classification_and_regression_metrics(data, predictions)
    print(evaluation_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    main(load_config(args.config))
```

This script assumes that you have the necessary FATE and Chainer libraries installed, and it uses a YAML configuration file named `config.yaml` by default. You should adjust the script according to your specific needs and the exact API usage in your FATE environment.