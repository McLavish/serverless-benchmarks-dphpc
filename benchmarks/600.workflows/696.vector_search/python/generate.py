import uuid
import numpy as np


def make_vector(embedding_dim: int, seed: int) -> list[float]:
    """
    Create a deterministic random embedding vector so runs are reproducible.
    Returned as a Python list to be JSON-serializable.
    """
    rng = np.random.default_rng(seed=seed)
    vec = rng.normal(size=embedding_dim).astype(np.float32)
    return vec.tolist()


def handler(event):
    """
    Input (from input.py):
      {
        "n_queries": int,
        "embedding_dim": int,
        "top_k": int,
        "request_id": optional
      }

    Output:
      {
        "queries": [
          {
            "query_id": "...",
            "vector": [...],
            "top_k": int,
            "request_id": "..."
          },
          ...
        ]
      }
    """
    n_queries = int(event["n_queries"])
    embedding_dim = int(event["embedding_dim"])
    top_k = int(event.get("top_k", max(1, min(5, n_queries))))
    request_id = event.get("request_id", str(uuid.uuid4())[:8])

    queries = []
    for idx in range(n_queries):
        vec = make_vector(embedding_dim, seed=idx)
        queries.append(
            {
                "query_id": f"query-{request_id}-{idx}",
                "vector": vec,
                "top_k": top_k,
                "request_id": request_id,
            }
        )

    return {"queries": queries}
