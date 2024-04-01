import numpy as np
from transformers import AutoTokenizer, AutoModel

from src.baselines.model.embed_baseline_model import EmbedBaseline


class CodeT5Baseline(EmbedBaseline):

    def __init__(self, pretrained_path: str,
                 device: str = "cuda",
                 checkpoint: str = "Salesforce/codet5p-110m-embedding"):
        super().__init__(pretrained_path, None)
        self.device = device
        self.checkpoint = checkpoint

    @staticmethod
    def name():
        return 'codet5'

    def embed(self, file_contents: np.ndarray[str], batch_size: int = 1) -> np.ndarray[float]:
        dumped_embeddings = self.load_embeddings()
        if dumped_embeddings is not None:
            assert len(file_contents) == dumped_embeddings.shape[0]
            return dumped_embeddings

        # For now, we do not finetune model
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.checkpoint, trust_remote_code=True).to(self.device)
        embeddings = []

        for i in range(0, len(file_contents), batch_size):
            inputs = tokenizer(file_contents[i: (i + batch_size)],
                               return_tensors="pt",
                               padding='max_length',
                               truncation=True,
                               return_attention_mask=False).to(self.device)
            batch_embeddings = model(**inputs)
            if self.device != 'cpu':
                batch_embeddings = batch_embeddings.to('cpu')

            embeddings.append(batch_embeddings.detach().numpy())
            del inputs
            del batch_embeddings

        embeddings = np.concatenate(embeddings)
        assert len(file_contents) == embeddings.shape[0]

        self.dump_embeddings(embeddings)
        print(f'Embeddings length: {len(embeddings[0])}')

        return embeddings
