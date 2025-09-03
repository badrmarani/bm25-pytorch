# BM25 in PyTorch

A minimal implementation of the [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) algorithm using PyTorch sparse tensors. You can use any tokenizer from HuggingFace, and you can run it on GPU.

```bash
pip install git+https://github.com/badrmarani/bm25-pytorch
```

## Usage

```python
from bm25pt import BM25

corpus = [
    "hello there good man!",
    "it is quite windy in london",
    "how is the weather today?"
]
queries = ["man", "windy london", "imagine"]

# device_id=0, 1, ..., or None for "cpu"
retriever = BM25(device_id=0, tokenizer_name_or_path="bert-base-uncased")
retriever.index(corpus)

print(retriever.score(queries))
# Output:
# tensor([[0.9330, 0.5578, 0.5864],
#         [1.0366, 1.7089, 0.9812],
#         [0.1997, 0.1842, 0.1916]])
```

Instead of refitting the retriever from scratch, you can update it with new documents as often as you need.

```python
# e.g., add new documents.
retriever.index(["wind is not the worst weather we could imagine"])
print(retriever.score(queries))
# Output:
# tensor([[1.0811, 0.6254, 0.6566, 0.5714],
#         [1.2211, 2.0658, 1.1583, 1.0035],
#         [0.6911, 0.6254, 0.6566, 0.9342]])
```
