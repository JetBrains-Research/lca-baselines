```python
import torch
from torch.utils.data import DataLoader, Dataset
from torchrec.distributed import DistributedModelParallel
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embeddings import EmbeddingBagCollection
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torch.optim import SGD
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ModuleSharder
from torch.distributed import init_process_group

class RandomDataset(Dataset):
    def __init__(self, num_embeddings, batch_size):
        self.num_embeddings = num_embeddings
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        x = torch.randint(0, self.num_embeddings, (10,))
        y = torch.randint(0, 2, (1,))
        return x, y

def _get_random_dataloader(num_embeddings, batch_size, pin_memory):
    dataset = RandomDataset(num_embeddings, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory)
    return dataloader

def train(num_embeddings, embedding_dim, dense_arch_layer_sizes, over_arch_layer_sizes, learning_rate):
    init_process_group(backend="nccl")
    device = torch.device("cuda")
    rank = torch.distributed.get_rank()
    backend = "nccl"

    embedding_configs = [EmbeddingBagConfig(name="test_embedding", embedding_dim=embedding_dim, num_embeddings=num_embeddings)]
    dlrm = DLRM(
        embedding_bag_collection=EmbeddingBagCollection(
            embedding_bag_configs=embedding_configs,
        ),
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
    )

    optimizer = SGD(dlrm.parameters(), lr=learning_rate)
    optimizer = KeyedOptimizerWrapper(optimizer)

    model = DistributedModelParallel(
        module=dlrm,
        device_ids=[device],
        broadcast_buffers=False,
        sharders=[],
    )

    dataloader = _get_random_dataloader(num_embeddings, 1024, True)

    for epoch in range(10):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = output.loss()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train(num_embeddings=10000, embedding_dim=128, dense_arch_layer_sizes=[512, 256, 128], over_arch_layer_sizes=[1024, 512, 1], learning_rate=0.01)
```