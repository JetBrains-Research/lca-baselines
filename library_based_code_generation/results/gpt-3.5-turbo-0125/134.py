import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.cluster import KMeans
from lightly.models import MemoryBankModule
from lightly.transforms import SMoGTransform
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

class SMoGModel(nn.Module):
    def __init__(self, backbone):
        super(SMoGModel, self).__init__()
        self.backbone = backbone
        self.projection_head = nn.Linear(512, 128)
        self.prediction_head = nn.Linear(128, 10)
    
    def cluster_features(self, features):
        kmeans = KMeans(n_clusters=10, random_state=0).fit(features)
        return kmeans.labels_
    
    def reset_group_features(self):
        pass
    
    def reset_momentum_weights(self):
        pass
    
    def forward(self, x):
        features = self.backbone(x)
        encoded = self.projection_head(features)
        predicted = self.prediction_head(encoded)
        return encoded, predicted

backbone = models.resnet18(pretrained=False)
model = SMoGModel(backbone)
memory_bank = MemoryBankModule(128, 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = SMoGTransform()
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    model.reset_momentum_weights()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        encoded, predicted = model(inputs)
        
        model.reset_group_features()
        
        loss = criterion(predicted, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        memory_bank.update_memory_bank(encoded)
    
    print(f'Epoch {epoch+1}, Average Loss: {loss.item()}')