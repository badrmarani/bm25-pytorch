import math
from collections import defaultdict

from tokenizers import Tokenizer


class BM25:
    def __init__(
        self,
        documents: list[str],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if tokenizer is None:
            raise ValueError()
        self.tokenizer = tokenizer
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.num_documents = len(documents)

        self.word_doc_freq = defaultdict(int)
        for document in documents:
            seen_word = set()
            for word in self.tokenizer.encode(document).tokens:
                if word not in seen_word:
                    self.word_doc_freq[word] += 1
                    seen_word.add(word)

        self.avgdl = sum([len(self.tokenizer.encode(document)) for document in documents]) / self.num_documents

    def idf(self, word: str) -> float:
        nq = self.word_doc_freq.get(word, 0)
        if nq == 0:
            return 0
        return math.log((self.num_documents - nq + 0.5) / (nq + 0.5) + 1)

    def score(self, queries: list[str]) -> list[list[float]]:
        _scores = [[]] * len(queries)
        for i, query in enumerate(queries):
            _scores[i] = [0.0] * self.num_documents
            for j, document in enumerate(self.documents):
                for word in self.tokenizer.encode(query).tokens:
                    f = self.tokenizer.encode(document).tokens.count(word)
                    a = f * (self.k1 + 1)
                    b = f + self.k1 * (1 + self.b * (len(self.tokenizer.encode(document)) - 1) / self.avgdl)
                    _scores[i][j] += self.idf(word) * a / b
        return _scores


if __name__ == "__main__":
    corpus = [
        "A high weight in tf-idf is reached by a high term frequency",
        "(in the given document) and a low document frequency of the term",
        "in the whole collection of documents; the weights hence tend to filter",
        "out common terms. Since the ratio inside the idf's log function is always",
        "greater than or equal to 1, the value of idf (and tf-idf) is greater than or equal",
        "to 0. As a term appears in more documents, the ratio inside the logarithm approaches",
        "1, bringing the idf and tf-idf closer to 0.",
    ]
    query = "common terms"

    tokenizer = Tokenizer.from_pretrained("t5-base")
    retriever = BM25(corpus, tokenizer=tokenizer)
    print(retriever.score([query]))
