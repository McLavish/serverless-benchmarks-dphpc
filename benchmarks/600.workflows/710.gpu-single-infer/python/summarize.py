def handler(event):
    return {
        "top1_idx": event["top1_idx"],
        "top1_score": event["top1_score"],
        "image_path": event["image_path"],
    }
