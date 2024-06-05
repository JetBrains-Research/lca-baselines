  import fate

def main(config_file):
    # Load the configuration file
    config = fate.config.load(config_file)

    # Create a pipeline for the binary classification task
    pipeline = fate.pipeline.Pipeline()

    # Add components to the pipeline
    pipeline.add(fate.data.DataReader(config['data_reader']))
    pipeline.add(fate.data.DataTransformer(config['data_transformer']))
    pipeline.add(fate.data.DataIntersector(config['data_intersector']))
    pipeline.add(fate.model.HeteroNN(config['hetero_nn']))
    pipeline.add(fate.model.GuestBottomModel(config['guest_bottom_model']))
    pipeline.add(fate.model.InteractiveLayer(config['interactive_layer']))
    pipeline.add(fate.model.GuestTopModel(config['guest_top_model']))
    pipeline.add(fate.model.Evaluator(config['evaluator']))

    # Compile and fit the pipeline
    pipeline.compile()
    pipeline.fit()

    # Print the summary of the HeteroNN component
    print(pipeline.get_component('hetero_nn').summary())

if __name__ == '__main__':
    main('config.yaml')