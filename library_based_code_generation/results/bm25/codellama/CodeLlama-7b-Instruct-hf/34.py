  import os
import sys
from fate_flow.pipelined_model import PipelinedModel
from fate_flow.pipelined_model import Pipeline
from fate_flow.pipelined_model import Role
from fate_flow.pipelined_model import Data
from fate_flow.pipelined_model import DataType
from fate_flow.pipelined_model import DataHeader
from fate_flow.pipelined_model import DataAlignment
from fate_flow.pipelined_model import HeteroFeatureBinningGuest
from fate_flow.pipelined_model import FeatureSelectionParam
from fate_flow.pipelined_model import HeteroFeatureSelection
from fate_flow.pipelined_model import HeteroFeatureBinningHost
from fate_flow.pipelined_model import _classification_and_regression_extract
from fate_flow.pipelined_model import FeatureBinningConverter
from fate_flow.pipelined_model import FeatureBinningParam
from fate_flow.pipelined_model import HomoFeatureBinning
from fate_flow.pipelined_model import HeteroFeatureBinning
from fate_flow.pipelined_model import test_feature_binning
from fate_flow.pipelined_model import BaseFeatureBinning
from fate_flow.pipelined_model import set_feature
from fate_flow.pipelined_model import quantile_binning_and_count
from fate_flow.pipelined_model import hetero_feature_binning_guest_runner
from fate_flow.pipelined_model import hetero_feature_binning_host_runner
from fate_flow.pipelined_model import _evaluate_classification_and_regression_metrics
from fate_flow.pipelined_model import hetero_feature_selection_runner
from fate_flow.pipelined_model import hetero_feature_selection_param

# Define the pipeline
pipeline = Pipeline()

# Define the data reading and transformation steps
data_reading_guest = Data(name="data_reading_guest", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)
data_reading_host = Data(name="data_reading_host", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)
data_transformation_guest = Data(name="data_transformation_guest", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)
data_transformation_host = Data(name="data_transformation_host", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)

# Define the intersection step
intersection = Data(name="intersection", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)

# Define the feature scaling step
feature_scaling = Data(name="feature_scaling", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)

# Define the feature binning step
feature_binning = Data(name="feature_binning", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)

# Define the data statistics step
data_statistics = Data(name="data_statistics", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)

# Define the Pearson correlation step
pearson_correlation = Data(name="pearson_correlation", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)

# Define the one-hot encoding step
one_hot_encoding = Data(name="one_hot_encoding", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)

# Define the feature selection step
feature_selection = Data(name="feature_selection", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)

# Define the logistic regression step
logistic_regression = Data(name="logistic_regression", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)

# Define the evaluation step
evaluation = Data(name="evaluation", data_type=DataType.DATA, data_header=DataHeader.NO_HEADER, data_alignment=DataAlignment.NO_ALIGNMENT)

# Define the roles for the pipeline
guest = Role(name="guest", data=data_reading_guest, data_transformation=data_transformation_guest, intersection=intersection, feature_scaling=feature_scaling, feature_binning=feature_binning, data_statistics=data_statistics, pearson_correlation=pearson_correlation, one_hot_encoding=one_hot_encoding, feature_selection=feature_selection, logistic_regression=logistic_regression, evaluation=evaluation)
host = Role(name="host", data=data_reading_host, data_transformation=data_transformation_host, intersection=intersection, feature_scaling=feature_scaling, feature_binning=feature_binning, data_statistics=data_statistics, pearson_correlation=pearson_correlation, one_hot_encoding=one_hot_encoding, feature_selection=feature_selection, logistic_regression=logistic_regression, evaluation=evaluation)
arbiter = Role(name="arbiter", data=data_reading_guest, data_transformation=data_transformation_guest, intersection=intersection, feature_scaling=feature_scaling, feature_binning=feature_binning, data_statistics=data_statistics, pearson_correlation=pearson_correlation, one_hot_encoding=one_hot_encoding, feature_selection=feature_selection, logistic_regression=logistic_regression, evaluation=evaluation)

# Define the pipeline components
pipeline.add_component(guest)
pipeline.add_component(host)
pipeline.add_component(arbiter)

# Define the pipeline parameters
pipeline_parameters = {
    "data_reading_guest": {
        "data_path": "data/guest.csv"
    },
    "data_reading_host": {
        "data_path": "data/host.csv"
    },
    "data_transformation_guest": {
        "transform_param": {
            "transform_type": "standardization"
        }
    },
    "data_transformation_host": {
        "transform_param": {
            "transform_type": "standardization"
        }
    },
    "intersection": {
        "intersect_param": {
            "intersect_type": "inner"
        }
    },
    "feature_scaling": {
        "scaling_param": {
            "scaling_type": "standardization"
        }
    },
    "feature_binning": {
        "binning_param": {
            "bin_num": 10
        }
    },
    "data_statistics": {
        "statistics_param": {
            "statistics_type": "summary"
        }
    },
    "pearson_correlation": {
        "correlation_param": {
            "correlation_type": "pearson"
        }
    },
    "one_hot_encoding": {
        "one_hot_param": {
            "one_hot_type": "binary"
        }
    },
    "feature_selection": {
        "feature_selection_param": {
            "feature_selection_type": "mutual_information"
        }
    },
    "logistic_regression": {
        "logistic_regression_param": {
