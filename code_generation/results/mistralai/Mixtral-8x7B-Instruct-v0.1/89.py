 ```python
import json
import numpy as np
from fate_flow.settings import BASE_DIR
from fate_flow.utils.serializable import loads as fd_loads, dumps as fd_dumps
from fate_flow.job_manager import JobManager
from fate_flow.entity.job import Job
from fate_flow.entity.model import Model
from fate_flow.entity.table import Table
from fate_flow.entity.data_profile import DataProfile
from fate_flow.entity.summary import Summary
from fate_flow.component.dataio.table_reader import TableReader
from fate_flow.component.dataio.table_writer import TableWriter
from fate_flow.component.transformer.polynomial_transformation import PolynomialTransformation
from fate_flow.component.transformer.min_max_scaler import MinMaxScaler
from fate_flow.component.estimator.logistic_regression import LogisticRegression
from fate_flow.component.evaluator.metric import Metric
from fate_flow.component.deploy.serving import Serving

def create_pipeline(config):
    job_manager = JobManager()

    # Create tables for guest and host
    guest_data_path = config['guest_data_path']
    host_data_path = config['host_data_path']
    guest_table = Table(name=config['guest_table_name'], data_path=guest_data_path)
    host_table = Table(name=config['host_table_name'], data_path=host_data_path)

    # Create data reader for guest and host
    guest_data_reader = TableReader(table=guest_table)
    host_data_reader = TableReader(table=host_table)

    # Create transformation component
    transformation = PolynomialTransformation(degree=config['degree'])

    # Create scaling component
    scaler = MinMaxScaler()

    # Create logistic regression model with specific parameters
    logistic_regression = LogisticRegression(penalty=config['penalty'],
                                             optimizer=config['optimizer'],
                                             tolerance=config['tolerance'],
                                             alpha=config['alpha'],
                                             max_iter=config['max_iter'],
                                             early_stopping=config['early_stopping'],
                                             batch_size=config['batch_size'],
                                             learning_rate=config['learning_rate'],
                                             decay=config['decay'],
                                             init_method=config['init_method'],
                                             random_seed=config['random_seed'],
                                             calibration_data_size=config['calibration_data_size'])

    # Create evaluation component
    evaluation = Metric(metric=config['metric'], top_k=config['top_k'])

    # Create serving component
    serving = Serving()

    # Create training pipeline
    train_pipeline = Job(name=config['train_pipeline_name'], components=[guest_data_reader,
                                                                          host_data_reader,
                                                                          transformation,
                                                                          scaler,
                                                                          logistic_regression],
                         description=config['train_pipeline_description'])

    # Create prediction pipeline
    predict_pipeline = Job(name=config['predict_pipeline_name'], components=[guest_data_reader,
                                                                            transformation,
                                                                            scaler,
                                                                            serving],
                          description=config['predict_pipeline_description'])

    # Compile and fit training pipeline
    job_manager.create_job(job=train_pipeline)
    job_manager.start_job(job_id=train_pipeline.job_id)
    job_manager.wait_job(job_id=train_pipeline.job_id)

    # Deploy selected components
    job_manager.deploy(job_id=train_pipeline.job_id,
                       components=[logistic_regression, serving],
                       model_name=config['model_name'],
                       model_version=config['model_version'])

    # Compile and fit prediction pipeline
    job_manager.create_job(job=predict_pipeline)
    job_manager.start_job(job_id=predict_pipeline.job_id)
    job_manager.wait_job(job_id=predict_pipeline.job_id)

    # Save DSL and configuration as JSON files
    with open(f"{BASE_DIR}/dsl/train_pipeline_dsl.json", "w") as f:
        f.write(fd_dumps(train_pipeline.to_dsl()))
    with open(f"{BASE_DIR}/config/train_pipeline_config.json", "w") as f:
        f.write(fd_dumps(train_pipeline.to_config()))

    # Print summaries of logistic regression and evaluation components
    logistic_regression_summary = Summary(job_id=train_pipeline.job_id,
                                          component_id=logistic_regression.component_id)
    evaluation_summary = Summary(job_id=train_pipeline.job_id,
                                 component_id=evaluation.component_id)
    print(json.dumps(fd_loads(job_manager.get_summary(summary=logistic_regression_summary)), indent=2))
    print(json.dumps(fd_loads(job_manager.get_summary(summary=evaluation_summary)), indent=2))

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f)
    create_pipeline(config)
```