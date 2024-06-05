```python
import argparse
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.components.component_base import ComponentBase
from fate_flow.components.dataio import DataIO
from fate_flow.components.intersection import Intersection
from fate_flow.components.feature_scale import FeatureScale
from fate_flow.components.hetero_feature_binning import HeteroFeatureBinning
from fate_flow.components.hetero_feature_selection import HeteroFeatureSelection
from fate_flow.components.hetero_logistic_regression import HeteroLogisticRegression
from fate_flow.components.evaluation import Evaluation
from fate_flow.components.one_hot_encoder import OneHotEncoder
from fate_flow.components.pearson import Pearson
from fate_flow.components.data_statistics import DataStatistics
from fate_flow.utils.config_adapter import JobRuntimeConfigAdapter

# Parse command line arguments for configuration file
parser = argparse.ArgumentParser(description='Run a machine learning pipeline using FATE.')
parser.add_argument('--config', type=str, help='Path to the configuration file', default='default_config.yaml')
args = parser.parse_args()

# Load configuration
RuntimeConfig.init_config(WORK_MODE=1)
config = JobRuntimeConfigAdapter(args.config)

# Define parameters for feature binning, feature selection, and logistic regression
feature_binning_params = {
    "method": "quantile",
    "bin_num": 5
}

feature_selection_params = {
    "method": "threshold",
    "threshold": 0.1
}

logistic_regression_params = {
    "penalty": "L2",
    "optimizer": "sgd",
    "batch_size": 320,
    "learning_rate": 0.15,
    "max_iter": 100
}

# Initialize components
data_io_guest = DataIO(name="data_io_guest", role="guest")
data_io_host = DataIO(name="data_io_host", role="host")
intersection = Intersection(name="intersection")
feature_scale_guest = FeatureScale(name="feature_scale_guest", role="guest")
feature_scale_host = FeatureScale(name="feature_scale_host", role="host")
hetero_feature_binning_guest = HeteroFeatureBinning(name="hetero_feature_binning_guest", role="guest", **feature_binning_params)
hetero_feature_binning_host = HeteroFeatureBinning(name="hetero_feature_binning_host", role="host", **feature_binning_params)
data_statistics = DataStatistics(name="data_statistics")
pearson = Pearson(name="pearson")
one_hot_encoder = OneHotEncoder(name="one_hot_encoder")
hetero_feature_selection = HeteroFeatureSelection(name="hetero_feature_selection", **feature_selection_params)
hetero_logistic_regression = HeteroLogisticRegression(name="hetero_logistic_regression", **logistic_regression_params)
evaluation = Evaluation(name="evaluation")

# Compile pipeline
pipeline = [
    data_io_guest, data_io_host, intersection,
    feature_scale_guest, feature_scale_host,
    hetero_feature_binning_guest, hetero_feature_binning_host,
    data_statistics, pearson, one_hot_encoder,
    hetero_feature_selection, hetero_logistic_regression, evaluation
]

# Fit pipeline
for component in pipeline:
    if isinstance(component, ComponentBase):
        component.run(config)
```