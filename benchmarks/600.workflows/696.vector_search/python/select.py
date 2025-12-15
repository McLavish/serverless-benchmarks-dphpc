import numpy as np


def score_vector(vec) -> float:
    """
    Simple scoring function for a query vector.
    Here we just use the L2 norm as a toy 'relevance' score.
    vec is a list[float] from generate.py.
    """
    v = np.asarray(vec, dtype=np.float32)
    return float(np.linalg.norm(v))


def handler(query):
    """
    Map-style handler: processes a single query.

    Input (one element from generate.py's "queries" list):
      {
        "query_id": str,
        "vector": list[float],
        "top_k": int,
        "request_id": str
      }

    Output (one element in the aggregated "queries" list for evaluate.py):
      {
        "query_id": str,
        "score": float,
        "top_k": int
      }
    """
    vec = query["vector"]
    score = score_vector(vec)

    return {
        "query_id": query["query_id"],
        "score": score,
        "top_k": query.get("top_k", 1),
    }
