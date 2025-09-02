from typing import Any, Sequence

import torch
from transformers import AutoTokenizer


def get_device(device_id: int | None = None) -> torch.device:
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
        if device_id is not None and 0 <= device_id < torch.cuda.device_count():
            device_name = f"cuda:{device_id}"
    return torch.device(device_name)


def generate_batch(data: Sequence[Any], batch_size: int):
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        yield batch


class BM25:
    def __init__(
        self,
        device_id: int | None = None,
        tokenizer_name_or_path: str | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b

        self.device = get_device(device_id)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
        self.tokenizer_fn = lambda s: tokenizer(s, return_tensors="pt", padding=True).input_ids.to(device=self.device)
        self.vocab_size = tokenizer.vocab_size

    def index(self, corpus: list[str], batch_size: int = 6) -> torch.Tensor:
        x = []
        self.total_num_documents = 0
        for batch in generate_batch(corpus, min(batch_size, len(corpus))):
            ids = self.tokenizer_fn(batch)
            num_documents, seq_length = ids.size()

            indices = torch.stack((torch.arange(num_documents, device=self.device).unsqueeze(1).expand(-1, seq_length), ids))

            _x = torch.sparse_coo_tensor(
                indices=indices.reshape(2, -1),
                values=(ids > 0).int().flatten(),
                size=(num_documents, self.vocab_size),
                device=self.device,
            )
            x.append(_x)
            self.total_num_documents += num_documents
        self.vs = torch.cat(x).coalesce()
        self.document_length = self.vs.sum(dim=1).to_dense()
        self.average_document_length = self.document_length.float().mean()
        return self.vs

    def idf(self, token_ids: torch.Tensor) -> torch.Tensor:
        nq = (self.vs.index_select(1, token_ids.flatten()).to_dense() > 0).sum(dim=0).view(token_ids.size())
        return torch.where(
            nq > 0,
            torch.log((self.total_num_documents - nq + 0.5) / (nq + 0.5) + 1),
            torch.zeros_like(token_ids),
        )

    def score(self, queries: str | list[str]) -> torch.Tensor:
        token_ids = self.tokenizer_fn(queries)
        f = self.vs.index_select(1, token_ids.flatten()).to_dense().view(self.total_num_documents, *token_ids.size())

        a = f * self.k1 + 1
        b = f + self.k1 * (1 + self.b * (self.document_length.view(-1, 1, 1) - 1) / self.average_document_length)
        return (self.idf(token_ids).unsqueeze(0) * a / b).sum(dim=2).t()
