 import argparse
import avalanche as av
import avalanche.models as avm
import torch
from avalanche.benchmarks.classic import MNIST
from avalanche.training.strategies import Replay
from avalanche.training.templates import OnlineEvaluation
from avalanche.data_loaders.pytorch_pl import PyTorchDataLoader
from avalanche.models.pytorch import SimpleMLP

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    return parser.parse\_args()

def main(args):
 devices = torch.device(f'cuda:{args.gpu}' if torch.cuda.is\_available() else 'cpu')

 transforms = av.transforms.Compose([
 av.transforms.ToTensor(),
 av.transforms.Normalize(mean=(0.1307,), std=(0.3081,))
 ])

 mnist_benchmark = MNIST(
 train\_classes=10, test\_classes=10, transform=transforms, download=True
 )

 strategy = Replay(
 model=SimpleMLP(input\_size=28 * 28, num\_classes=10),
 data\_loader=PyTorchDataLoader,
 batch\_size=32,
 devices=devices,
 strategy\_args={
 'num\_workers': 2,
 'storage\_policy': av.storage_policies.ReservoirSamplingBuffer(
 max\_size=1000
 ),
 'callbacks': [
 av.callbacks.LearningRateTracker(
 monitor='accuracy',
 factor=0.1,
 patience=2,
 min\_lr=1e-5
 ),
 av.callbacks.ModelCheckpoint(
 filepath='checkpoint.pth',
 monitor='accuracy',
 save\_top\_k=1,
 mode='max'
 ),
 ]
 },
)

 train\_stream, test\_stream = mnist_benchmark.train\_test\_streams(
 memory\_buffer\_size=100,
 shuffle=True,
 seed=1,
 stream_class_sampling=av.utils.stream_sampling.RoundRobinClassSampling(10)
 )

 metrics = [
 av.metrics.AverageMetric(av.metrics.Accuracy()),
 av.metrics.AverageMetric(av.metrics.Loss())
 ]

 for epoch in range(5):
 print(f'Epoch: {epoch}')
 strategy.train(train\_stream, epoch)
 strategy.eval(test\_stream, OnlineEvaluation(metrics))

 results = [res for res in strategy.results]

 for res in results:
 print(res)

if **name** == '**main**':
 args = parse\_arguments()
 main(args)