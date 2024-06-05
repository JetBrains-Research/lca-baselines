```python
import argparse
import json
from fate_arch.common import log
from fate_flow.entity.types import JobStatus
from fate_flow.operation.job_saver import JobSaver
from fate_flow.operation.job_tracker import Tracker
from fate_flow.settings import stat_logger
from fate_flow.utils.job_utils import generate_job_id
from pipeline.backend.config import Backend, PIPELINE_MODEL
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform, Evaluation, FeatureScale, HomoLR, Reader
from pipeline.interface.model import Model
from pipeline.interface.data import Data

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Define the pipeline
    pipeline = PipeLine(backend=Backend.EGGROLL, initiator_role='guest', initiator_party_id=config['guest_party_id'])

    # Add Reader for Guest
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=config['guest_party_id']).component_param(table={"name": config['guest_table_name'], "namespace": config['guest_table_namespace']})
    
    # Add Reader for Host
    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='host', party_id=config['host_party_id']).component_param(table={"name": config['host_table_name'], "namespace": config['host_table_namespace']})

    # Data Transform
    data_transform_0 = DataTransform(name="data_transform_0")

    # Feature Scaling
    feature_scale_0 = FeatureScale(name="feature_scale_0")

    # Logistic Regression Model
    homo_lr_0 = HomoLR(name="homo_lr_0", penalty=config['penalty'], optimizer=config['optimizer'], tol=config['tolerance'], alpha=config['alpha'], max_iter=config['max_iterations'], early_stop=config['early_stopping'], batch_size=config['batch_size'], learning_rate=config['learning_rate'], decay=config['decay'], init_param={'init_method': config['init_method']}, cv_param={'n_splits': config['cv_n_splits'], 'shuffle': config['cv_shuffle']})

    # Evaluation
    evaluation_0 = Evaluation(name="evaluation_0")

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data), model=Model(data_transform_0.output.model))
    pipeline.add_component(feature_scale_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(homo_lr_0, data=Data(train_data=feature_scale_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=homo_lr_0.output.data))

    # Compile pipeline
    pipeline.compile()

    # Fit pipeline
    pipeline.fit()

    # Deploy selected components
    pipeline.deploy_component([data_transform_0, homo_lr_0])

    # Create prediction pipeline
    predict_pipeline = pipeline.copy()
    predict_pipeline.add_component(reader_0)
    predict_pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    predict_pipeline.add_component(homo_lr_0, data=Data(data=data_transform_0.output.data))

    # Compile prediction pipeline
    predict_pipeline.compile()

    # Make predictions
    predict_pipeline.predict()

    # Save DSL and configuration of the prediction pipeline
    with open("prediction_pipeline_dsl.json", "w") as f:
        json.dump(predict_pipeline.get_dsl(), f)
    with open("prediction_pipeline_config.json", "w") as f:
        json.dump(predict_pipeline.get_config(), f)

    # Print summaries
    print("Logistic Regression Summary:")
    print(pipeline.get_component("homo_lr_0").get_summary())
    print("Evaluation Summary:")
    print(pipeline.get_component("evaluation_0").get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config_path)
```