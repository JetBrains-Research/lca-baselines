```python
from fate_flow import init
from fate_flow import pipeline
from fate_flow import job_parameters

init.init_config(job_parameters)
pipeline.run(pipeline.create_pipeline(
    data=[
        {"name": "data_reading", "namespace": "guest"},
        {"name": "data_reading", "namespace": "host"},
        {"name": "data_transformation", "namespace": "guest"},
        {"name": "data_transformation", "namespace": "host"},
        {"name": "intersection"},
        {"name": "feature_scaling"},
        {"name": "feature_binning", "params": {"binning_param": {"method": "quantile"}}},
        {"name": "data_statistics"},
        {"name": "pearson_correlation"},
        {"name": "one_hot_encoding"},
        {"name": "feature_selection", "params": {"selection_param": {"method": "L1"}}},
        {"name": "logistic_regression", "params": {"penalty": "l2"}},
        {"name": "evaluation"}
    ],
    roles={
        "guest": ["data_reading", "data_transformation", "feature_scaling", "feature_binning"],
        "host": ["data_reading", "data_transformation", "feature_scaling", "feature_binning"],
        "arbiter": ["data_statistics", "pearson_correlation", "one_hot_encoding", "feature_selection", "logistic_regression", "evaluation"]
    }
))
```