def handler(event):
    """
    Reduce-style handler: takes all scored queries and picks the top-k.

    Input:
      {
        "queries": [
          {
            "query_id": str,
            "score": float,
            "top_k": int
          },
          ...
        ]
      }

    Output:
      {
        "top_queries": [
          { "query_id": str, "score": float, "top_k": int },
          ...
        ]
      }
    """
    results = event.get("queries", [])
    if not results:
        return {"top_queries": []}

    top_k = int(results[0].get("top_k", len(results)))
    top_queries = sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]

    return {"top_queries": top_queries}
