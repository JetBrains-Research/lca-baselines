 import torch
import fate
from torch import nn
from torch.optim import Adam
from torch.nn.functional import binary\_cross\_entropy

def create\_guest\_bottom\_model():
return nn.Sequential(
nn.Linear(10, 8),
nn.ReLU(),
nn.Linear(8, 1)
)

def create\_guest\_top\_model():
return nn.Sequential(
nn.Linear(10, 8),
nn.ReLU(),
nn.Linear(8, 1)
)

def create\_host\_bottom\_model():
return nn.Sequential(
nn.Linear(10, 8),
nn.ReLU(),
nn.Linear(8, 1)
)

def create\_interactive\_layer():
return nn.Sequential(
nn.Linear(3, 4),
nn.ReLU(),
nn.Linear(4, 1)
)

def create\_hetero\_nn(guest\_bottom\_model, guest\_top\_model, host\_bottom\_model, interactive\_layer):
hetero\_nn = fate.HeteroNN()
hetero\_nn.set\_bottom_model('guest', guest\_bottom\_model)
hetero\_nn.set\_bottom_model('host', host\_bottom\_model)
hetero\_nn.set\_interaction_model('guest\_host', interactive\_layer)
hetero\_nn.set\_top_model('guest', guest\_top\_model)
return hetero\_nn

def compile\_and\_fit\_hetero\_nn(hetero\_nn, train\_data, epochs, learning\_rate, batch\_size):
optimizer = Adam(hetero\_nn.parameters(), lr=learning\_rate)
loss\_fn = binary\_cross\_entropy
hetero\_nn.compile(optimizer=optimizer, loss=loss\_fn, metrics=['accuracy'])
hetero\_nn.fit(train\_data, epochs=epochs, batch\_size=batch\_size)
return hetero\_nn

def print\_hetero\_nn\_summary(hetero\_nn):
hetero\_nn.summary()

def main(config\_file):
config = fate.Config(config\_file)
reader = fate.Reader(config)
data\_transform = fate.DataTransform(config)
train\_data = data\_transform.transform(reader.read())
guest\_bottom\_model = create\_guest\_bottom\_model()
guest\_top\_model = create\_guest\_top\_model()
host\_bottom\_model = create\_host\_bottom\_model()
interactive\_layer = create\_interactive\_layer()
hetero\_nn = create\_hetero\_nn(guest\_bottom\_model, guest\_top\_model, host\_bottom\_model, interactive\_layer)
hetero\_nn = compile\_and\_fit\_hetero\_nn(hetero\_nn, train\_data, epochs=10, learning\_rate=0.01, batch\_size=32)
print\_hetero\_nn\_summary(hetero\_nn)