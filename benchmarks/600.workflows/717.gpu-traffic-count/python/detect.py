import json
import logging
import os
import shutil
import uuid
from collections import Counter

import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

from . import storage

logger = logging.getLogger(__name__)
client = storage.storage.get_instance()

_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
_model = fasterrcnn_resnet50_fpn(weights=_weights).cuda().eval()
_transform = _weights.transforms()
_categories = _weights.meta["categories"]

_vehicle_labels = {name for name in _categories if name in {"car", "truck", "bus", "motorcycle", "bicycle"}}


def handler(event):
    input_bucket = event["input_bucket"]
    output_bucket = event["output_bucket"]
    benchmark_bucket = event["benchmark_bucket"]
    frames = event["segments"]
    prefix = event["prefix"]

    tmp_dir = os.path.join("/tmp", str(uuid.uuid4()))
    os.makedirs(tmp_dir, exist_ok=True)

    results = []
    try:
        for frame in frames:
            local_img = os.path.join(tmp_dir, frame)
            client.download(benchmark_bucket, f"{input_bucket}/{frame}", local_img)

            img = Image.open(local_img).convert("RGB")
            x = _transform(img).to("cuda")
            with torch.inference_mode():
                out = _model([x])[0]

            labels = [int(l) for l in out["labels"].tolist()]
            scores = out["scores"].tolist()
            counts = Counter()
            detections = []
            for lbl, score in zip(labels, scores):
                if score < 0.4:
                    continue
                name = _categories[lbl]
                if name not in _vehicle_labels:
                    continue
                counts[name] += 1
                detections.append({"label": name, "score": float(score)})

            result = {
                "frame": frame,
                "vehicles": {k: int(v) for k, v in counts.items()},
                "total": int(sum(counts.values())),
            }

            local_json = os.path.join(tmp_dir, f"{os.path.splitext(frame)[0]}_det.json")
            with open(local_json, "w", encoding="utf-8") as f:
                json.dump(result, f)

            client.upload(
                benchmark_bucket,
                f"{output_bucket}/{prefix}{os.path.basename(local_json)}",
                local_json,
                unique_name=False,
            )

            results.append(result)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {**event, "results": results}
