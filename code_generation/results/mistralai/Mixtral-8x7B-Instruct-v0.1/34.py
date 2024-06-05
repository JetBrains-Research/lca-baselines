 ```python
import argparse
import sys
from fate_flow import FateFlow
from fate_flow.entity.data_flow import TableProcessItem
from fate_flow.entity.pipeline import PipelineItem
from fate_flow.entity.role_info import RoleInfo
from fate_flow.protobuf.meta import PipelineCreateRequest

def create_pipeline(role, config_file=None):
    if config_file is None:
        config_file = "config.json"

    data_transformation_guest = TableProcessItem(
        table="data_table",
        operations=["data_processing"]
    )

    data_transformation_host = TableProcessItem(
        table="data_table",
        operations=["data_processing"]
    )

    intersection = TableProcessItem(
        table="intersection_table",
        operations=["intersection"]
    )

    feature_scaling_guest = TableProcessItem(
        table="scaling_table",
        operations=["feature_scaling"]
    )

    feature_scaling_host = TableProcessItem(
        table="scaling_table",
        operations=["feature_scaling"]
    )

    feature_binning_params = {
        "feature_binning_guest": {
            "features": ["feature1", "feature2"],
            "bins": 5
        },
        "feature_binning_host": {
            "features": ["feature1", "feature2"],
            "bins": 5
        }
    }

    feature_selection_params = {
        "feature_selection_guest": {
            "method": "chi2",
            "top_k": 10
        },
        "feature_selection_host": {
            "method": "chi2",
            "top_k": 10
        }
    }

    logistic_regression_params = {
        "logistic_regression_guest": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs"
        },
        "logistic_regression_host": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs"
        }
    }

    pipeline = PipelineItem(
        name="ml_pipeline",
        role_infos=[
            RoleInfo(role="guest", table_process_items=[data_transformation_guest, feature_scaling_guest, feature_binning_params["feature_binning_guest"], feature_selection_params["feature_selection_guest"], "logistic_regression_guest"]),
            RoleInfo(role="host", table_process_items=[data_transformation_host, feature_scaling_host, feature_binning_params["feature_binning_host"], feature_selection_params["feature_selection_host"], "logistic_regression_host"]),
            RoleInfo(role="arbiter", table_process_items=["data_statistics", "pearson_correlation", "one_hot_encoding"])
        ]
    )

    request = PipelineCreateRequest()
    request.pipeline = pipeline
    request.role = role
    request.component_param = {
        "data_statistics": {},
        "pearson_correlation": {},
        "one_hot_encoding": {},
        "feature_binning_guest": feature_binning_params["feature_binning_guest"],
        "feature_binning_host": feature_binning_params["feature_binning_host"],
        "feature_selection_guest": feature_selection_params["feature_selection_guest"],
        "feature_selection_host": feature_selection_params["feature_selection_host"],
        "logistic_regression_guest": logistic_regression_params["logistic_regression_guest"],
        "logistic_regression_host": logistic_regression_params["logistic_regression_host"]
    }

    fate_flow = FateFlow()
    fate_flow.create_pipeline(request)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--config_file", type=str, default=None)
    args = parser.parse_args()

    create_pipeline(args.role, args.config_file)
```