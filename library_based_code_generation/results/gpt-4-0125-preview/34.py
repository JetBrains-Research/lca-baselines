```python
import argparse
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_arch.common import profile
from fate_flow.settings import stat_logger
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform, Intersection, FeatureScale, FeatureBinning, DataStatistics, Pearson, OneHotEncoder, FeatureSelection, HomoLR, Evaluation
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.interface.model import Model
from pipeline.runtime.entity import JobParameters

# Parse command line arguments for configuration file
parser = argparse.ArgumentParser(description='Run a machine learning pipeline using FATE.')
parser.add_argument('--config', type=str, help='Path to the configuration file', default='default_config.yaml')
args = parser.parse_args()

# Define feature binning, feature selection, and logistic regression parameters
feature_binning_params = {
    "method": "quantile",
    "bin_num": 5
}

feature_selection_params = {
    "method": "threshold",
    "threshold": 0.1
}

logistic_regression_params = {
    "optimizer": "sgd",
    "batch_size": 320,
    "learning_rate": 0.15,
    "early_stop": "diff",
    "max_iter": 100
}

# Initialize pipeline
pipeline = PipeLine()

# Add reader component for guest and host
pipeline.add_component(Reader(name="reader_guest"), role='guest')
pipeline.add_component(Reader(name="reader_host"), role='host')

# Data transformation for guest and host
pipeline.add_component(DataTransform(name="data_transform_guest"), data=Data(data=reader_guest.output.data), role='guest')
pipeline.add_component(DataTransform(name="data_transform_host"), data=Data(data=reader_host.output.data), role='host')

# Intersection component
pipeline.add_component(Intersection(name="intersection"), data=Data(data=data_transform_guest.output.data, data1=data_transform_host.output.data))

# Feature scaling for guest and host
pipeline.add_component(FeatureScale(name="feature_scale_guest"), data=Data(data=intersection.output.data), role='guest')
pipeline.add_component(FeatureScale(name="feature_scale_host"), data=Data(data=intersection.output.data1), role='host')

# Feature binning
pipeline.add_component(FeatureBinning(name="feature_binning", **feature_binning_params), data=Data(data=feature_scale_guest.output.data))

# Data statistics
pipeline.add_component(DataStatistics(name="data_statistics"), data=Data(data=feature_binning.output.data))

# Pearson correlation
pipeline.add_component(Pearson(name="pearson"), data=Data(data=feature_binning.output.data))

# One-hot encoding
pipeline.add_component(OneHotEncoder(name="one_hot_encoder"), data=Data(data=pearson.output.data))

# Feature selection
pipeline.add_component(FeatureSelection(name="feature_selection", **feature_selection_params), data=Data(data=one_hot_encoder.output.data))

# Logistic regression
pipeline.add_component(HomoLR(name="homo_lr", **logistic_regression_params), data=Data(train_data=feature_selection.output.data))

# Evaluation
pipeline.add_component(Evaluation(name="evaluation"), data=Data(data=homo_lr.output.data))

# Compile pipeline
pipeline.compile()

# Fit pipeline
pipeline.fit(JobParameters(backend=0, work_mode=0, conf=args.config))
```