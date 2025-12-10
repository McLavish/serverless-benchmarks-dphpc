import numpy as np


def apply_threshold_segmentation(scan: np.ndarray, n_organs: int) -> np.ndarray:
    """
    Simple multi-threshold segmentation to identify different tissue types.
    Simulates GPU-accelerated semantic segmentation (e.g., U-Net inference).
    """
    segmentation = np.zeros_like(scan, dtype=np.int32)

    # Create threshold ranges for different organs/tissues
    thresholds = np.linspace(0, 1, n_organs + 1)

    for organ_id in range(1, n_organs + 1):
        lower = thresholds[organ_id - 1]
        upper = thresholds[organ_id]
        mask = (scan >= lower) & (scan < upper)
        segmentation[mask] = organ_id

    return segmentation


def compute_organ_stats(segmentation: np.ndarray, n_organs: int) -> dict:
    """Compute statistics for each segmented organ."""
    stats = {}
    total_pixels = segmentation.size

    for organ_id in range(1, n_organs + 1):
        mask = segmentation == organ_id
        pixel_count = mask.sum()

        stats[f"organ_{organ_id}"] = {
            "pixel_count": int(pixel_count),
            "percentage": float(pixel_count / total_pixels * 100),
        }

    return stats


def detect_anomalies(segmentation: np.ndarray, scan: np.ndarray) -> list:
    """
    Detect potential anomalies (bright spots, lesions, etc.).
    Simulates GPU-accelerated anomaly detection.
    """
    anomalies = []

    # Detect bright regions that could indicate lesions or tumors
    threshold = 0.75
    anomaly_mask = scan > threshold

    if anomaly_mask.any():
        # Find connected components (simplified)
        coords = np.argwhere(anomaly_mask)
        if len(coords) > 0:
            # Calculate centroid
            centroid_y, centroid_x = coords.mean(axis=0)
            anomalies.append({
                "type": "bright_region",
                "location": [float(centroid_y), float(centroid_x)],
                "size": int(len(coords)),
                "severity": float(scan[anomaly_mask].mean()),
            })

    return anomalies


def handler(schedule):
    slice_id = schedule["slice_id"]
    scan_data = np.array(schedule["scan_data"], dtype=np.float32)
    height = schedule["height"]
    width = schedule["width"]
    n_organs = schedule["n_organs"]

    # Perform segmentation (simulating GPU inference)
    segmentation = apply_threshold_segmentation(scan_data, n_organs)

    # Compute organ statistics
    organ_stats = compute_organ_stats(segmentation, n_organs)

    # Detect anomalies
    anomalies = detect_anomalies(segmentation, scan_data)

    return {
        "slice_id": slice_id,
        "slice_idx": schedule["slice_idx"],
        "organ_stats": organ_stats,
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
    }
