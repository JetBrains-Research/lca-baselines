 Here is a simplified example of how you might set up a pipeline using FATE (Fairness, Accountability, and Transparency) library for a multi-party computation scenario. Note that this is a basic example and might need adjustments based on your specific use case.

```python
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from fate_flow.dsl import Pipeline, DataReader, Transformer, Scaler, Trainer, Evaluator, Deployer, Predictor
from fate_flow.common.parameter import Config

def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Define data readers for guest and host
    guest_data_reader = DataReader(config['guest_data_source'])
    host_data_reader = DataReader(config['host_data_source'])

    # Define data transformers
    transformer = Transformer(config['transformer'])

    # Define feature scaler
    scaler = Scaler(config['scaler'])

    # Define logistic regression model
    logistic_regression = LogisticRegression(
        penalty=config['logistic_regression']['penalty'],
        optimizer=config['logistic_regression']['optimizer'],
        tolerance=config['logistic_regression']['tolerance'],
        max_iter=config['logistic_regression']['max_iter'],
        class_weight=config['logistic_regression']['class_weight'],
        solver=config['logistic_regression']['solver'],
        multi_class='ovr',
        random_state=config['logistic_regression']['random_state'],
        n_jobs=config['logistic_regression']['n_jobs'],
        C=config['logistic_regression']['C'],
        l1_ratio=config['logistic_regression']['l1_ratio'],
        dual=config['logistic_regression']['dual'],
        fit_intercept=True,
        warm_start=False
    )

    # Define cross-validation
    cv = StratifiedKFold(n_splits=config['cv']['n_splits'], shuffle=True, random_state=config['cv']['random_state'])

    # Define training pipeline
    training_pipeline = Pipeline(
        [
            (guest_data_reader.name, guest_data_reader),
            (host_data_reader.name, host_data_reader),
            (transformer.name, transformer),
            (scaler.name, scaler),
            (Trainer(logistic_regression).name, Trainer(logistic_regression)),
            (Evaluator().name, Evaluator())
        ]
    )

    # Define deployment
    deployer = Deployer(config['deployer'])

    # Compile and fit the training pipeline
    training_pipeline.compile(cv=cv)
    training_pipeline.fit(guest_data_reader.read(), host_data_reader.read())

    # Deploy the trained model
    deployed_model = deployer.deploy(training_pipeline.get_model())

    # Define prediction pipeline
    prediction_pipeline = Pipeline([(guest_data_reader.name, guest_data_reader), (deployed_model.name, deployed_model)])

    # Compile and save the prediction pipeline
    prediction_pipeline.compile()
    prediction_pipeline_config = prediction_pipeline.to_config()
    with open('prediction_pipeline.json', 'w') as f:
        json.dump(prediction_pipeline_config, f)

    # Create a predictor from the prediction pipeline
    predictor = Predictor(prediction_pipeline)

    # Make predictions
    X = np.random.rand(10, config['data']['features'])
    y = np.random.randint(2, size=10)
    predictions = predictor.predict(X)

    # Print summaries
    logistic_regression.fit(X, y).get_params()
    training_pipeline.get_component(Trainer(logistic_regression).name).get_summary()
    training_pipeline.get_component(Evaluator().name).get_summary()

if __name__ == "__main__":
    main('config.json')
```

This code assumes that you have a configuration file named `config.json` that contains the necessary parameters for your specific use case. The configuration file should be structured according to the keys used in the code. For example:

```json
{
    "guest_data_source": "guest_data_reader",
    "host_data_source": "host_data_reader",
    "transformer": "StandardScaler",
    "scaler": "StandardScaler",
    "logistic_regression": {
        "penalty": "l2",
        "optimizer": "sag",
        "tolerance": 1e-4,
        "max_iter": 100,
        "class_weight": "balanced",
        "solver": "sag",
        "random_state": 42,
        "n_jobs": 1,
        "C": 1.0,
        "l1_ratio": 0.1,
        "dual": False,
        "fit_intercept": True,
        "warm_start": False
    },
    "cv": {
        "n_splits": 5,
        "random_state": 42
    },
    "deployer": "LocalDeployer",
    "data": {
        "features": 10
    }
}
```