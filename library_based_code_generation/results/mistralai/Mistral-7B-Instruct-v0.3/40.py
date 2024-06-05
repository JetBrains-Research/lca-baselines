 Here is the Python code using the nni library to perform the tasks as described:

```python
import torch
import torchvision
import torchvision.transforms as transforms
from nni.retiarii.api import get_run_id
from nni.retiarii.utils import get_worker_id
from nni.parameters import OneOf, IntegerParam, ListParam, DictParam
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet
from nni.nas.pytorch import NASBench101SearchSpace

model = ResNet18()
criterion = torch.nn.CrossEntropyLoss()
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5]))
])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

def train(model, optimizer, scheduler, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    print('Epoch: %d Loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

def evaluate(model):
    model.eval()
    running_corrects = 0
    running_total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_corrects += (predictions == labels).sum().item()
    print('Accuracy: %.3f' % (running_corrects.float() / running_total))

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in range(30):
    train(model, optimizer, scheduler, epoch)

fine_tuned_model = model

teacher_model = fine_tuned_model.clone()

pruner_config = {
    'pruner_type': 'TaylorPruner',
    'target_ops': ['Conv2d'],
    'sparse_ratio': 0.5,
    'num_steps': 100,
    'num_runs': 30
}
pruner = nni.nas.pytorch.Pruner(fine_tuned_model, pruner_config)
pruner.fit(train_loader, criterion, epochs=1)

quantizer_config = {
    'quantizer_type': 'QATQuantizer',
    'target_ops': ['Conv2d', 'BatchNorm2d'],
    'bit_width': 8,
    'start_step': 100
}
quantizer = nni.nas.pytorch.Quantizer(fine_tuned_model, quantizer_config)
quantizer.fit(train_loader, criterion, epochs=1)

distiller_config = {
    'distiller_type': 'DynamicLayerwiseDistiller',
    'target_ops': ['Conv2d'],
    'loss_type': 'MSE',
    'teacher_model': teacher_model
}
distiller = nni.nas.pytorch.Distiller(fine_tuned_model, distiller_config)
compressed_model = distiller.compress(fine_tuned_model, train_loader, criterion, epochs=60)

speedup_config = {
    'speedup_type': 'ModelSpeedup',
    'pruner': pruner
}
speedup_module = nni.nas.pytorch.ModelSpeedup(compressed_model, speedup_config)
speedup_model = speedup_module.apply(compressed_model)

evaluate(speedup_model)

quantization_config = {
    'quantizer_type': 'Quantizer',
    'target_ops': ['Conv2d', 'BatchNorm2d'],
    'calibration_data': train_loader
}
nni.nas.pytorch.quantize(speedup_model, quantization_config)

quantized_model = speedup_model
evaluate(quantized_model)
```

This code fine-tunes a ResNet18 model on the Cifar10 dataset for 30 epochs, creates a teacher model, prunes the model using TaylorPruner and AGPPruner, quantizes the model using QATQuantizer, distills the model using DynamicLayerwiseDistiller, speeds up the model using ModelSpeedup, and evaluates the accuracy of the compressed and quantized model.