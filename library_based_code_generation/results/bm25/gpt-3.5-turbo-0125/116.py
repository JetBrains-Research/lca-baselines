```python
# Define hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Create learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Define main function
def main():
    # Initialize ResNet18 model
    model = torchvision.models.resnet18(pretrained=True)
    
    # Define normalization and transformation operations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Set up logging to Tensorboard, text file, and stdout
    writer = SummaryWriter()
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler('training.log'))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    # Define evaluation plugin with various metrics
    eval_plugin = EvaluationPlugin(metrics=['accuracy', 'precision', 'recall', 'f1'])
    
    # Set seed value and create CLEAR benchmark
    seed = 42
    benchmark = CLEARBenchmark(seed=seed)
    
    # Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Define continual learning strategy using Naive method
    cl_strategy = Naive(model, optimizer, criterion, train_mb_size=10, train_epochs=1, eval_mb_size=10, device=device)
    
    # Training loop
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        cl_strategy.eval(benchmark.test_stream)
        
        # Save model after each experience
        torch.save(model.state_dict(), f'model_{experience.current_experience}.pt')
    
    # Generate accuracy matrix and compute CLEAR metrics
    accuracy_matrix = cl_strategy.accuracy_matrix(benchmark.test_stream)
    clear_metrics = benchmark.evaluate(cl_strategy, benchmark.test_stream)
    
    # Log results to text file
    with open('results.txt', 'w') as f:
        f.write(f'Accuracy Matrix: {accuracy_matrix}\n')
        f.write(f'CLEAR Metrics: {clear_metrics}\n')
```