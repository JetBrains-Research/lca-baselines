import fate
import torch

def main(config_file):
    pipeline = fate.Pipeline()
    
    reader = fate.Reader()
    data_transform = fate.DataTransform()
    intersection = fate.Intersection()
    
    hetero_nn = fate.HeteroNN(epochs=10, learning_rate=0.001, batch_size=32, task_type='binary_classification')
    
    guest_bottom_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU()
    )
    guest_top_model = torch.nn.Sequential(
        torch.nn.Linear(20, 1),
        torch.nn.Sigmoid()
    )
    host_bottom_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU()
    )
    interactive_layer = torch.nn.Linear(20, 10)
    
    hetero_nn.add_guest_bottom_model(guest_bottom_model)
    hetero_nn.add_guest_top_model(guest_top_model)
    hetero_nn.add_host_bottom_model(host_bottom_model)
    hetero_nn.add_interactive_layer(interactive_layer)
    
    hetero_nn.compile(optimizer='adam', loss='binary_crossentropy')
    
    evaluation = fate.Evaluation()
    
    pipeline.add_component(reader)
    pipeline.add_component(data_transform)
    pipeline.add_component(intersection)
    pipeline.add_component(hetero_nn)
    pipeline.add_component(evaluation)
    
    pipeline.compile()
    pipeline.fit(training_data)
    
    hetero_nn.summary()

if __name__ == '__main__':
    config_file = 'config.json'
    main(config_file)