from bm25pt.bm25 import BM25

if __name__ == "__main__":
    corpus = [
        "Hello there good man!",
        "It is quite windy in London London",
        "How is the weather today?",
    ]
    queries = ["man", "windy london"]

    retriever = BM25(tokenizer_name_or_path="bert-base-uncased")
    retriever.index(corpus)

    print(retriever.score(queries))
