import torch

# Tiny embedding MLP to keep dependencies light.
_proj = torch.nn.Sequential(
    torch.nn.Linear(32, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
).cuda()


def _hash_text(text):
    h = torch.zeros(32, device="cuda", dtype=torch.float32)
    for i, ch in enumerate(text.encode("utf-8")):
        h[i % 32] += float(ch)
    return h


def handler(event):
    texts = event["texts"]
    batch = torch.stack([_hash_text(t) for t in texts])
    with torch.inference_mode():
        embs = _proj(batch).detach().cpu().numpy().tolist()
    return {"embeddings": embs, "count": len(texts)}
