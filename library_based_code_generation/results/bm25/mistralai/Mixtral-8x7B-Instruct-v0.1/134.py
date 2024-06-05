 import torch
import torch.nn as nn
import torchvision.transforms as transforms
from lightly.models.modules import MemoryBankModule
from sklearn.cluster import KMeans
from torchvision.datasets import CIFAR10
from lightly.models.resnet import resnet18

class SMoGModel(nn.Module):
def **init**(self, backbone):
super(SMoGModel, self).**init**()
self.backbone = backbone
self.projection_head = nn.Sequential(
nn.Linear(512, 512),
nn.ReLU(),
nn.Linear(512, 128)
)
self.prediction_head = nn.Linear(128, 10)
self.kmeans = KMeans(32)

def reset_group_features(self):
self.backbone._reset_group_features()

def reset_momentum_weights(self):
self.backbone._reset_momentum_weights()

def cluster_features(self, features):
self.kmeans.fit(features.detach().cpu().numpy())
return torch.from_numpy(self.kmeans.cluster_centers_).to(features.device)

def forward(self, x):
features = self.backbone(x)
encoded = self.projection_head(features)
predicted = self.prediction_head(encoded)
return encoded, predicted

backbone = resnet18(pretrained=False, num_classes=0)
model = SMoGModel(backbone)
memory_bank = MemoryBankModule(128, 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
memory_bank = memory_bank.to(device)

transform = transforms.Compose([SMoGTransform()])
dataset = CIFAR10(root="data", train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):
model.backbone.momentum_weight_decay_update()
for inputs, _ in dataloader:
inputs = inputs.to(device)
model.zero_grad()
encoded, predicted = model(inputs)
group_features = model.cluster_features(encoded)
model.backbone.update_group_features(group_features)
memory_bank = model.backbone.memory_bank_update(memory_bank, encoded)
loss = criterion(predicted, inputs.detach().clone().long().to(device))
loss.backward()
optimizer.step()
print(f"Epoch: {epoch+1}, Loss: {loss.item()}")