# BM25 in PyTorch

A minimal implementation of the [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) algorithm using PyTorch sparse tensors. You can use any tokenizer from Hugging Face, and you can run the operations on the GPU.

```bash
pip install git+https://github.com/badrmarani/bm25-pytorch
```

## Usage

```python
from bm25pt.bm25 import BM25

corpus = [
    "Hello there good man!",
    "It is quite windy in London London",
    "How is the weather today?"
]
queries = ["man", "windy london"]

device_id = 0 # cuda:0
retriever = BM25(device_id=device_id, tokenizer_name_or_path="bert-base-uncased")
retriever.index(corpus)

print(retriever.score(queries))
# Output:
# tensor([[0.9330, 0.5578, 0.5864],
#         [1.0366, 1.7089, 0.9812]])
```
