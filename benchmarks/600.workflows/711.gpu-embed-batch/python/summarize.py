import math


def handler(event):
    embs = event["embeddings"]
    norms = [math.sqrt(sum(x * x for x in v)) for v in embs]
    return {"count": event["count"], "avg_norm": sum(norms) / len(norms)}
