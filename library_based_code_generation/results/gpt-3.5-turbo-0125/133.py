import fate
from fate.utils import load_config

def main(config_file):
    config = load_config(config_file)

    pipeline = fate.Pipeline()
    reader = fate.components.Reader(name='reader', **config['reader_params'])
    transformer = fate.components.Transformer(name='transformer', **config['transformer_params'])
    intersect = fate.components.Intersect(name='intersect', **config['intersect_params'])
    hetero_nn = fate.components.HeteroNN(name='hetero_nn', **config['hetero_nn_params'])
    evaluator = fate.components.Evaluator(name='evaluator', **config['evaluator_params'])

    pipeline.add_component(reader)
    pipeline.add_component(transformer)
    pipeline.add_component(intersect)
    pipeline.add_component(hetero_nn)
    pipeline.add_component(evaluator)

    pipeline.compile()
    pipeline.fit()

    hetero_nn.summary()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()

    main(args.config)