 Here is a simplified Python code example using the FATE library for the given task. Please note that this is a basic example and might need adjustments based on your specific use case and FATE library version.

```python
import json
import os
import pickle
import numpy as np
from fate_arch.dsl import PipelineConfig, PipelineJob, PredictPipeline, make_temp_pipeline, parse_a_tree
from fate_arch.common import fit_a_learner, guest_traverse_a_tree, host_traverse_a_tree, _classification_and_regression_extract, lr_train_pipeline
from fate_arch.common import _run_pipeline_jobs, sbt_train_pipeline, is_a_quadratic_residue, init_data_and_variable, data_and_header_alignment, _evaluate_classification_and_regression_metrics
from fate_arch.common import decompress_and_unpack, create_and_get, pack_and_encrypt

def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Initialize data and variables
    init_data_and_variable(config['data_path'])

    # Create and configure the pipeline
    pipeline_config = PipelineConfig()
    pipeline_config.name = config['pipeline_name']
    pipeline_config.meta = PipelineConfigMeta(**config['pipeline_meta'])

    # Data reader
    data_reader = parse_a_tree(config['data_reader'])

    # Data transformer
    data_transformer = parse_a_tree(config['data_transformer'])

    # Feature scaler
    feature_scaler = parse_a_tree(config['feature_scaler'])

    # Logistic regression model
    lr_config = config['logistic_regression']
    lr_pipeline = lr_train_pipeline(lr_config)

    # Training pipeline
    training_pipeline = make_temp_pipeline()
    training_pipeline.add_component(data_reader)
    training_pipeline.add_component(data_transformer)
    training_pipeline.add_component(feature_scaler)
    training_pipeline.add_component(lr_pipeline)

    # Multi-party computation setup
    training_pipeline.set_guest_traversable_components([data_reader, data_transformer, feature_scaler])
    training_pipeline.set_host_traversable_components([lr_pipeline])

    # Compile and fit the training pipeline
    training_pipeline_config = PipelineConfig(pipeline=training_pipeline)
    training_pipeline_job = PipelineJob(config=training_pipeline_config)
    _run_pipeline_jobs([training_pipeline_job])

    # Deploy selected components
    deployed_components = [data_reader, feature_scaler]

    # Prediction pipeline
    prediction_pipeline = PredictPipeline()
    prediction_pipeline.add_component(data_reader)
    prediction_pipeline.add_component(deployed_components[0])
    prediction_pipeline.add_component(deployed_components[1])

    # Compile and use the prediction pipeline
    prediction_pipeline_config = PipelineConfig(pipeline=prediction_pipeline)
    prediction_pipeline_job = PipelineJob(config=prediction_pipeline_config)
    prediction_pipeline = pack_and_encrypt(prediction_pipeline_job)

    # Make predictions
    X = np.array([[1.0, 2.0, 3.0]])  # Replace with your test data
    y_pred = prediction_pipeline.predict(X)
    print("Predictions:", y_pred)

    # Save the DSL and configuration of the prediction pipeline as JSON files
    prediction_pipeline_config_json = prediction_pipeline_config.to_json()
    with open('prediction_pipeline_config.json', 'w') as f:
        json.dump(prediction_pipeline_config_json, f, indent=4)

    # Save the serialized prediction pipeline
    with open('prediction_pipeline.pickle', 'wb') as f:
        pickle.dump(prediction_pipeline, f)

    # Print summaries of the logistic regression and evaluation components
    logistic_regression_summary = _classification_and_regression_extract(lr_pipeline)
    evaluation_summary = _evaluate_classification_and_regression_metrics(training_pipeline)
    print("Logistic Regression Summary:", logistic_regression_summary)
    print("Evaluation Summary:", evaluation_summary)

if __name__ == "__main__":
    main("config.json")
```

This code assumes that you have a `config.json` file with the following structure:

```json
{
    "data_path": "path/to/data",
    "pipeline_name": "my_pipeline",
    "pipeline_meta": {
        "guest": "Alice",
        "host": "Bob",
        "arbiter": "Charlie"
    },
    "data_reader": {
        "guest": {
            "table": "guest_table",
            "columns": ["column1", "column2"]
        },
        "host": {
            "table": "host_table",
            "columns": ["column1", "column2"]
        }
    },
    "data_transformer": {
        "guest": {
            "transforms": ["StandardScaler"]
        },
        "host": {
            "transforms": ["StandardScaler"]
        }
    },
    "feature_scaler": {
        "guest": {
            "transforms": ["StandardScaler"]
        },
        "host": {
            "transforms": ["StandardScaler"]
        }
    },
    "logistic_regression": {
        "penalty": "l2",
        "optimizer": "sgd",
        "tol": 1e-3,
        "alpha": 0.01,
        "max_iter": 100,
        "early_stopping": True,
        "batch_size": 32,
        "learning_rate": "constant",
        "decay": 0.001,
        "initialization": "uniform",
        "cross_validation": {
            "cv": 5,
            "random_state": 42
        }
    }
}
```