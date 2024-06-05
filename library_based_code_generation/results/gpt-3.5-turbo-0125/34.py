import argparse
from fate_flow.client import Clients
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.feature.feature_selection.feature_selection import FeatureSelection
from federatedml.linear_model.logistic_regression import LogisticRegression
from federatedml.util import consts

# Define feature binning parameters
feature_binning_param = {
    "method": "quantile",
    "params": {
        "bin_num": 10
    }
}

# Define feature selection parameters
feature_selection_param = {
    "method": "variance",
    "params": {
        "threshold": 0.5
    }
}

# Define logistic regression parameters
logistic_regression_param = {
    "penalty": "L2",
    "max_iter": 100
}

# Define pipeline
pipeline = [
    {
        "name": "data_io",
        "role": "guest",
        "initiator": {
            "role": "guest",
            "party_id": 100
        },
        "role_parameters": {
            "guest": {
                "data": {"name": "name1"},
                "label": {"name": "label1"}
            }
        }
    },
    {
        "name": "data_io",
        "role": "host",
        "initiator": {
            "role": "guest",
            "party_id": 100
        },
        "role_parameters": {
            "host": {
                "data": {"name": "name2"},
                "label": {"name": "label2"}
            }
        }
    },
    {
        "name": "intersection",
        "role": "guest"
    },
    {
        "name": "intersection",
        "role": "host"
    },
    {
        "name": "feature_scale",
        "role": "guest"
    },
    {
        "name": "feature_scale",
        "role": "host"
    },
    {
        "name": "feature_binning",
        "role": "guest",
        "params": feature_binning_param
    },
    {
        "name": "feature_binning",
        "role": "host",
        "params": feature_binning_param
    },
    {
        "name": "data_statistics"
    },
    {
        "name": "pearson_correlation"
    },
    {
        "name": "one_hot_encoding"
    },
    {
        "name": "feature_selection",
        "params": feature_selection_param
    },
    {
        "name": "logistic_regression",
        "params": logistic_regression_param
    },
    {
        "name": "evaluation"
    }
]

# Compile and fit pipeline
job_parameters = {
    "work_mode": 1
}
job_dsl = {
    "components": pipeline
}
job_runtime_conf = {
    "initiator": {
        "role": "guest",
        "party_id": 100
    }
}

job_id = Clients().submit_job(
    job_type="train",
    job_parameters=job_parameters,
    job_dsl=job_dsl,
    job_runtime_conf=job_runtime_conf
)

Clients().query_job(job_id=job_id)