# benchmarks/600.workflows/718.gpu-document-qna/input.py
import os

size_generators = {
    "test": (4, 500, 2),
    "small": (16, 1500, 3),
    "large": (48, 3000, 4),
}


def buckets_count():
    return (1, 1)


def generate_input(
    data_dir,
    size,
    benchmarks_bucket,
    input_buckets,
    output_buckets,
    upload_func,
    nosql_func,
):
    if data_dir is None:
        raise ValueError(
            "/path/to/documents/\n"
            "  └── docs/\n"
            "       ├── doc1.txt\n"
            "       ├── doc2.txt\n"
            "       └── ..."
        )

    num_docs, max_tokens, top_k = size_generators[size]

    docs_dir = os.path.join(data_dir, "docs")
    if not os.path.isdir(docs_dir):
        raise ValueError(f"docs dir not exist: {docs_dir}")

    doc_files = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".txt"))
    if not doc_files:
        raise ValueError(f"no txt files under dir: {docs_dir}")

    new_docs = []
    for i in range(num_docs):
        doc = doc_files[i % len(doc_files)]
        name = f"{i:08d}.txt"
        path = os.path.join(docs_dir, doc)
        new_docs.append(name)
        upload_func(0, name, path)

    return {
        "segments": new_docs,
        "benchmark_bucket": benchmarks_bucket,
        "input_bucket": input_buckets[0],
        "output_bucket": output_buckets[0],
        "max_tokens": max_tokens,
        "top_k": top_k,
        "question": "Summarize the key risks mentioned in the document.",
    }
