```python
import argparse
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_arch.common import file_utils
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform, FeatureScale, HomoLR, Evaluation
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.runtime.entity import JobParameters, Model
from pipeline.utils.tools import load_job_config

def main(config_file):
    # Load configuration
    conf = load_job_config(config_file)
    guest_data_table = conf['guest_data_table']
    host_data_table = conf['host_data_table']
    pipeline_config = conf['pipeline_config']

    # Define the pipeline
    pipeline = PipeLine()

    # Reader
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=conf['guest_id']).component_param(table=guest_data_table)
    reader_0.get_party_instance(role='host', party_id=conf['host_id']).component_param(table=host_data_table)

    # Data Transform
    data_transform_0 = DataTransform(name="data_transform_0")

    # Feature Scale
    feature_scale_0 = FeatureScale(name="feature_scale_0")

    # Homo Logistic Regression
    homo_lr_0 = HomoLR(name="homo_lr_0", penalty=pipeline_config['penalty'], optimizer=pipeline_config['optimizer'],
                       tol=pipeline_config['tolerance'], alpha=pipeline_config['alpha'],
                       max_iter=pipeline_config['max_iterations'], early_stop=pipeline_config['early_stopping'],
                       batch_size=pipeline_config['batch_size'], learning_rate=pipeline_config['learning_rate'],
                       decay=pipeline_config['decay'], init_param={'init_method': pipeline_config['init_method']},
                       cv_param=pipeline_config['cross_validation'])

    # Evaluation
    evaluation_0 = Evaluation(name="evaluation_0")

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(feature_scale_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(homo_lr_0, data=Data(train_data=feature_scale_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=homo_lr_0.output.data))

    # Compile and fit the pipeline
    pipeline.compile()
    pipeline.fit(JobParameters(backend=1, work_mode=1))

    # Deploy selected components
    pipeline.deploy_component([data_transform_0, feature_scale_0, homo_lr_0])

    # Create prediction pipeline
    predict_pipeline = PipeLine()
    predict_pipeline.add_component(reader_0)
    predict_pipeline.add_component(data_transform_0)
    predict_pipeline.add_component(feature_scale_0)
    predict_pipeline.add_component(homo_lr_0)
    predict_pipeline.compile()

    # Use the prediction pipeline
    predict_pipeline.predict(JobParameters(backend=1, work_mode=1))

    # Save DSL and configuration
    predict_pipeline.save_dsl(file_utils.get_project_base_directory() + '/predict_dsl.json')
    predict_pipeline.save_predict_conf(file_utils.get_project_base_directory() + '/predict_conf.json')

    # Print summaries
    print(homo_lr_0.get_summary())
    print(evaluation_0.get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config_file)
```