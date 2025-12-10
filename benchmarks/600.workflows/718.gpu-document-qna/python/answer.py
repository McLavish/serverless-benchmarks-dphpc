import json
import math
import os
import shutil
import uuid
from typing import List

import torch

from . import storage

client = storage.storage.get_instance()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def handler(event):
    segments = event["segments"]
    output_bucket = segments[0]["output_bucket"]
    benchmark_bucket = segments[0]["benchmark_bucket"]
    top_k = event.get("top_k", 3)

    tmp_dir = os.path.join("/tmp", str(uuid.uuid4()))
    os.makedirs(tmp_dir, exist_ok=True)

    scores: List[dict] = []
    try:
        for seg in segments:
            doc = seg["doc"]
            prefix = seg["prefix"]
            json_path = os.path.join(tmp_dir, f"{os.path.splitext(doc)[0]}_emb.json")
            client.download(
                benchmark_bucket,
                f"{output_bucket}/{prefix}{os.path.basename(json_path)}",
                json_path,
            )
            with open(json_path, "r", encoding="utf-8") as f:
                emb_data = json.load(f)
            doc_vec = torch.tensor(emb_data["embedding"])
            q_vec = torch.tensor(emb_data["question_embedding"])
            sim = _cosine(doc_vec, q_vec)
            scores.append({"doc": doc, "score": sim, "question": emb_data["question"]})

        top = sorted(scores, key=lambda x: x["score"], reverse=True)[: top_k or len(scores)]
        answer = {
            "question": top[0]["question"] if top else "",
            "top_docs": top,
            "total_docs": len(scores),
        }

        answer_path = os.path.join(tmp_dir, "answer.json")
        with open(answer_path, "w", encoding="utf-8") as f:
            json.dump(answer, f, indent=2)

        client.upload(
            benchmark_bucket,
            f"{output_bucket}/answer.json",
            answer_path,
            unique_name=False,
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return answer
