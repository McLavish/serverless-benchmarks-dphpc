import json
import os
import shutil
import uuid

import torch
from transformers import AutoModel, AutoTokenizer

from . import storage

client = storage.storage.get_instance()

_MODEL_NAME = "intfloat/e5-small-v2"
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModel.from_pretrained(_MODEL_NAME).cuda().eval()


def _encode(text: str) -> list:
    tokens = _tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    for k in tokens:
        tokens[k] = tokens[k].cuda()
    with torch.inference_mode():
        outputs = _model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze(0).cpu().tolist()


def handler(event):
    input_bucket = event["input_bucket"]
    output_bucket = event["output_bucket"]
    benchmark_bucket = event["benchmark_bucket"]
    doc = event["doc"]
    prefix = event["prefix"]
    question = event["question"]

    tmp_dir = os.path.join("/tmp", str(uuid.uuid4()))
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        local_doc = os.path.join(tmp_dir, doc)
        client.download(benchmark_bucket, f"{input_bucket}/{doc}", local_doc)
        with open(local_doc, "r", encoding="utf-8") as f:
            text = f.read()

        doc_vec = _encode(text)
        q_vec = _encode(question)

        result = {
            "doc": doc,
            "embedding": doc_vec,
            "question_embedding": q_vec,
            "question": question,
        }

        local_json = os.path.join(tmp_dir, f"{os.path.splitext(doc)[0]}_emb.json")
        with open(local_json, "w", encoding="utf-8") as f:
            json.dump(result, f)

        client.upload(
            benchmark_bucket,
            f"{output_bucket}/{prefix}{os.path.basename(local_json)}",
            local_json,
            unique_name=False,
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {**event, "embedding": doc_vec, "question_embedding": q_vec}
