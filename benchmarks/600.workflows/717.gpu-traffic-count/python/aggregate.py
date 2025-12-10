import json
import os
import shutil
import uuid
from collections import Counter

from . import storage

client = storage.storage.get_instance()


def _download_det(frame: str, tmp_dir: str, benchmark_bucket: str, output_bucket: str, prefix: str):
    base, _ = os.path.splitext(frame)
    remote_name = f"{prefix}{base}_det.json"
    local_json = os.path.join(tmp_dir, f"{base}_det.json")
    try:
        client.download(benchmark_bucket, f"{output_bucket}/{remote_name}", local_json)
        return local_json
    except Exception:
        return None


def handler(event):
    frames = event["segments"]
    benchmark_bucket = event["benchmark_bucket"]
    output_bucket = event["output_bucket"]
    prefix = event["prefix"]

    tmp_dir = os.path.join("/tmp", str(uuid.uuid4()))
    os.makedirs(tmp_dir, exist_ok=True)

    vehicle_totals = Counter()
    missing = []
    try:
        for frame in frames:
            det_path = _download_det(frame, tmp_dir, benchmark_bucket, output_bucket, prefix)
            if det_path and os.path.exists(det_path):
                with open(det_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                vehicle_totals.update(data.get("vehicles", {}))
            else:
                missing.append(frame)

        summary = {
            "total_frames": len(frames),
            "labeled_frames": len(frames) - len(missing),
            "vehicle_totals": {k: int(v) for k, v in vehicle_totals.items()},
            "missing": missing,
        }

        summary_path = os.path.join(tmp_dir, "traffic_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        client.upload(
            benchmark_bucket,
            f"{output_bucket}/{prefix}traffic_summary.json",
            summary_path,
            unique_name=False,
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {**event, "summary": summary}
