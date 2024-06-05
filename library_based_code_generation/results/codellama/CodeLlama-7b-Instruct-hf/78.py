  from fate.pipeline import Pipeline
from fate.components import (
    DataReader,
    DataTransformer,
    Sampler,
    FeatureBinner,
    OneHotEncoder,
    LogisticRegression,
    LocalBaselineModel,
    SecureBoosting,
    Evaluator
)

def main(config_file):
    # Load configuration file
    config = Pipeline.load_config(config_file)

    # Create pipeline
    pipeline = Pipeline(config)

    # Add components to pipeline
    pipeline.add(DataReader(config['data_reader']))
    pipeline.add(DataTransformer(config['data_transformer']))
    pipeline.add(Sampler(config['sampler']))
    pipeline.add(FeatureBinner(config['feature_binner']))
    pipeline.add(OneHotEncoder(config['one_hot_encoder']))
    pipeline.add(LogisticRegression(config['logistic_regression']))
    pipeline.add(LocalBaselineModel(config['local_baseline_model']))
    pipeline.add(SecureBoosting(config['secure_boosting']))
    pipeline.add(Evaluator(config['evaluator']))

    # Compile and fit pipeline
    pipeline.compile()
    pipeline.fit()

    # Print summary of evaluation components
    print(pipeline.evaluate())

if __name__ == '__main__':
    main('config.json')