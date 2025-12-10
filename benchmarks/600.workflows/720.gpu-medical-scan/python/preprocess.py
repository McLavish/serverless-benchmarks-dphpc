import uuid
import numpy as np


def normalize_slice(data: np.ndarray) -> np.ndarray:
    """Normalize CT/MRI slice to [0, 1] range."""
    min_val, max_val = data.min(), data.max()
    if max_val - min_val > 0:
        return (data - min_val) / (max_val - min_val)
    return data


def generate_synthetic_scan(height: int, width: int, seed: int) -> np.ndarray:
    """Generate synthetic medical scan slice with anatomical features."""
    rng = np.random.default_rng(seed=seed)

    # Create base tissue structure
    scan = rng.normal(0.3, 0.1, (height, width)).astype(np.float32)

    # Add circular organ-like structures
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2

    # Main organ (e.g., liver, kidney)
    radius = min(height, width) // 3
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    scan[mask] += rng.normal(0.4, 0.05, mask.sum()).astype(np.float32)

    # Add some smaller structures (lesions, vessels)
    for i in range(3):
        offset_x = rng.integers(-radius//2, radius//2)
        offset_y = rng.integers(-radius//2, radius//2)
        small_radius = rng.integers(10, 30)
        small_mask = (x - center_x - offset_x)**2 + (y - center_y - offset_y)**2 <= small_radius**2
        scan[small_mask] += rng.normal(0.2, 0.03, small_mask.sum()).astype(np.float32)

    return normalize_slice(scan)


def handler(event):
    n_slices = int(event["n_slices"])
    height = int(event["height"])
    width = int(event["width"])
    n_organs = int(event["n_organs"])
    scan_id = event.get("scan_id", str(uuid.uuid4())[:8])

    slices = []
    for idx in range(n_slices):
        # Generate synthetic scan data
        scan_data = generate_synthetic_scan(height, width, seed=idx)

        slices.append({
            "slice_id": f"slice-{scan_id}-{idx:03d}",
            "slice_idx": idx,
            "scan_data": scan_data.tolist(),  # Convert to list for JSON serialization
            "height": height,
            "width": width,
            "n_organs": n_organs,
            "scan_id": scan_id,
        })

    return {"slices": slices}
