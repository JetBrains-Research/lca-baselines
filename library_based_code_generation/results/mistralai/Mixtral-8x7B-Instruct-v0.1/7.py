 import argparse
import avalanche as av
import avalanche.models as avm
import avalanche.benchmarks as avb
import avalanche.training as avt
import avalanche.evaluation as avl
import avalanche.logging as avl_logging
import avalanche.metrics as avm
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0,
                    help='Device to use for computations.')
args = parser.parse\_args()
device = f'cuda:{args.device}' if torch.cuda.is\_available() else 'cpu'

transform\_train = av.transforms.Compose([
    av.transforms.ToTensor(),
    av.transforms.Normalize(mean=[0.1307], std=[0.3081])
])

transform\_test = av.transforms.Compose([
    av.transforms.ToTensor(),
    av.transforms.Normalize(mean=[0.1307], std=[0.3081]),
    av.transforms.RandomErasing(p=0.5, v=0.4, device=device)
])

benchmark = avb.MNIST(transforms={'train': transform\_train, 'test': transform\_test})

model = avm.MLP(
    input\_size=benchmark.num\_features,
    hidden\_size=128,
    num\_classes=benchmark.num\_classes,
    output\_activation=nn.LogSoftmax(dim=1)
)

model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.NLLLoss()

text\_logger = avl\_logging.TextLogger(output_dir='logs', name='text\_logger')
interactive\_logger = avl\_logging.InteractiveLogger()
csv\_logger = avl\_logging.CSVLogger(output_dir='logs', name='csv\_logger')
tensorboard\_logger = avl\_logging.TensorboardLogger(output_dir='logs', name='tensorboard\_logger')

loggers = [text\_logger, interactive\_logger, csv\_logger, tensorboard\_logger]

evaluator = avl.AvalancheEvaluator(
    loggers=loggers,
    metrics=[
        avm.Accuracy(),
        avm.Loss(),
        avm.ClassAccuracy(),
        avm.AMCA(),
        avm.Forgetting(),
        avm.BackwardTransfer(),
        avm.ForwardTransfer(),
        avm.CPUUsage(),
        avm.Timing(),
        avm.RAMUsage(),
        avm.GPUUsage(),
        avm.DiskUsage(),
        avm.MAC(),
        avm.LabelsRepartition()
    ]
)

strategy = avt.NaiveContinualStrategies(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=10,
    valid_mb_size=1000,
    evaluator=evaluator,
    device=device
)

for experience in benchmark.train\_stream:
    strategy.train(experience)
    strategy.eval(experience.dataset)
    print(evaluator.get\_metric\_dict())

print(evaluator.get\_metric\_dict())