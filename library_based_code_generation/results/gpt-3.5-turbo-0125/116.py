```python
# Define hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Create learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Define main function
def main():
    model = torchvision.models.resnet18(pretrained=True)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    loggers = [TensorboardLogger(), TextLogger(), StdOutLogger()]
    
    eval_plugin = EvaluationPlugin(metrics=[Accuracy(), F1Score()])
    
    seed = 42
    benchmark = CLEAR()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    strategy = Naive(model, optimizer, criterion)
    
    for experience in benchmark.train_stream:
        strategy.train(experience)
        model.save('model_checkpoint.pth')
        eval_plugin.evaluate(model, benchmark.test_stream)
    
    accuracy_matrix = benchmark.accuracy_matrix()
    clear_metrics = benchmark.clear_metrics()
    
    with open('results.txt', 'w') as f:
        f.write(f'Accuracy Matrix: {accuracy_matrix}\n')
        f.write(f'CLEAR Metrics: {clear_metrics}\n')
```