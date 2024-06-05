```python
import torch
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embeddings import EmbeddingBagCollection
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import torch.optim as optim
import random

class RandomDataset(Dataset):
    def __init__(self, num_embeddings, batch_size):
        self.num_embeddings = num_embeddings
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        return {
            "features": torch.randint(0, self.num_embeddings, (10,)),
            "labels": torch.rand((1,))
        }

def _get_random_dataloader(num_embeddings, batch_size, pin_memory):
    dataset = RandomDataset(num_embeddings, batch_size)
    return DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory)

def train(num_embeddings, embedding_dim, dense_architecture, over_architecture, learning_rate):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    device = torch.device("cuda")

    embedding_configs = [EmbeddingBagConfig(name=str(i), embedding_dim=embedding_dim, num_embeddings=num_embeddings) for i in range(num_embeddings)]
    model = EmbeddingBagCollection(embedding_configs=embedding_configs)
    model = DistributedModelParallel(model, device_ids=[dist.get_rank()], sharding_type=ShardingType.DATA_PARALLEL)
    model = DDP(model)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    dataloader = _get_random_dataloader(num_embeddings, 1024, True)

    for epoch in range(10):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch["features"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs)
            loss = (outputs - labels).pow(2).mean()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train(1000, 64, [1024, 512, 256], [256, 128], 0.01)
```