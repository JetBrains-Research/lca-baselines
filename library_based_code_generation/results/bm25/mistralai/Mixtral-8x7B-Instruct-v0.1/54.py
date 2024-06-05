 import argparse
import avalanche as av
import torch
from avalanche.benchmarks.classic import MNIST
from avalanche.models.simple_mlp import SimpleMLP
from avalanche.training.strategies import Replay
from avalanche.training.templates import VanillaTemplate
from avalanche.models import LayerAndParameter

parser = argparse.ArgumentParser()
parser.add\_argument('--gpu', type=int, default=0, help='GPU device to use')
args = parser.parse\_args()
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is\_available() else 'cpu')

transform = av.transforms.Compose([
av.transforms.ToTensor(),
av.transforms.Normalize(mean=[0.1307], std=[0.3081])
])

mnist\_train, mnist\_test = MNIST(root='./data', download=True, train=True, transform=transform), \
MNIST(root='./data', download=True, train=False, transform=transform)

train\_stream, test\_stream = av.data\_streams.Stream(mnist\_train, batch\_size=32, shuffle=True), \
av.data\_streams.Stream(mnist\_test, batch\_size=32, shuffle=False)

model = SimpleMLP(num_classes=10, input\_size=28 * 28)
model = model.to(device)

strategy = Replay(
model=model,
criterion=av.losses.CrossEntropyLoss(),
optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
train_mb_size=32,
valid_mb_size=32,
train_epochs=1,
device=device,
checkpoint\_dir='checkpoints',
)

scenario\_cfg = {
'num\_experiments': 1,
'evaluate\_every\_n\_epochs': 1,
'eval\_method': test\_ocl\_scenario\_stream,
'metrics': [av.metrics.Accuracy()],
'log\_dir': 'logs',
'scenario\_type': 'online',
'strategy': strategy,
'model': model,
'device': device,
'on\_train\_begin': lambda strategy, **kwargs: strategy.eval(),
'on\_train\_end': lambda strategy, **kwargs: strategy.train(),
}

scenario = av.scenarios.GenericScenario(train\_stream, test\_stream, **scenario\_cfg)

for epoch in range(10):
metrics = scenario.train(epoch)
print(f'Epoch: {epoch + 1}, ' + ', '.join(f'{name}: {value.mid:.3f}' for name, value in metrics.items()))

training\_results = scenario.eval(scenario.test\_stream)
print(f'Test accuracy over all tasks: {training\_results[0].mid:.3f}')