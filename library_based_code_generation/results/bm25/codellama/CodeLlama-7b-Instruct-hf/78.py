  Here is an example of Python code that creates a pipeline for a federated learning task using the FATE library:
```
import fate

def main(config_file):
    # Load the configuration file
    config = fate.PipelineConfig(config_file)

    # Create a pipeline for the guest party
    guest_pipeline = fate.PipelineJob(config, 'guest')

    # Create a pipeline for the host party
    host_pipeline = fate.PipelineJob(config, 'host')

    # Add components to the pipelines
    guest_pipeline.add_component(fate.parse_a_tree, 'read_data')
    guest_pipeline.add_component(fate.transform_data, 'transform_data')
    guest_pipeline.add_component(fate.sample, 'sample')
    guest_pipeline.add_component(fate.feature_binning, 'feature_binning')
    guest_pipeline.add_component(fate.one_hot_encoding, 'one_hot_encoding')
    guest_pipeline.add_component(fate.logistic_regression, 'logistic_regression')
    guest_pipeline.add_component(fate.local_baseline_model, 'local_baseline_model')
    guest_pipeline.add_component(fate.secure_boosting, 'secure_boosting')
    guest_pipeline.add_component(fate.evaluation, 'evaluation')

    host_pipeline.add_component(fate.parse_a_tree, 'read_data')
    host_pipeline.add_component(fate.transform_data, 'transform_data')
    host_pipeline.add_component(fate.sample, 'sample')
    host_pipeline.add_component(fate.feature_binning, 'feature_binning')
    host_pipeline.add_component(fate.one_hot_encoding, 'one_hot_encoding')
    host_pipeline.add_component(fate.logistic_regression, 'logistic_regression')
    host_pipeline.add_component(fate.local_baseline_model, 'local_baseline_model')
    host_pipeline.add_component(fate.secure_boosting, 'secure_boosting')
    host_pipeline.add_component(fate.evaluation, 'evaluation')

    # Compile and fit the pipelines
    guest_pipeline.compile()
    guest_pipeline.fit()
    host_pipeline.compile()
    host_pipeline.fit()

    # Print the summary of the evaluation components
    print(guest_pipeline.evaluation.summary())
    print(host_pipeline.evaluation.summary())

if __name__ == '__main__':
    main('config.json')
```
This code assumes that the configuration file is named `config.json` and is located in the same directory as the Python script. The configuration file should contain the necessary information for the pipelines, such as the data tables for the guest and host parties, the feature columns, and the hyperparameters for the components.

The `fate` module is used to access the FATE library and its components. The `PipelineJob` class is used to create a pipeline for a guest or host party, and the `add_component` method is used to add components to the pipeline. The `compile` and `fit` methods are used to compile and fit the pipelines, respectively. The `evaluation` component is used to evaluate the performance of the pipelines.

Note that this is just an example code and may need to be modified to fit the specific requirements of your federated learning task.