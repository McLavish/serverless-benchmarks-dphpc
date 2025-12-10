# benchmarks/600.workflows/717.gpu-traffic-count/input.py
import os

size_generators = {
    "test": (6, 3),
    "small": (40, 8),
    "large": (160, 16),
}


def buckets_count():
    return (1, 1)


def generate_input(
    data_dir,
    size,
    benchmarks_bucket,
    input_buckets,
    output_buckets,
    upload_func,
    nosql_func,
):
    if data_dir is None:
        raise ValueError(
            "/path/to/traffic_frames/\n"
            "  └── frames/\n"
            "       ├── cam01_0001.jpg\n"
            "       ├── cam01_0002.jpg\n"
            "       └── ..."
        )

    num_frames, batch_size = size_generators[size]

    frames_dir = os.path.join(data_dir, "frames")
    if not os.path.isdir(frames_dir):
        raise ValueError(f"frames dir not exist: {frames_dir}")

    frame_files = sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not frame_files:
        raise ValueError(f"no jpg/png under dir: {frames_dir}")

    new_frames = []
    for i in range(num_frames):
        frame = frame_files[i % len(frame_files)]
        ext = os.path.splitext(frame)[1].lower() or ".jpg"
        name = f"{i:08d}{ext}"
        path = os.path.join(frames_dir, frame)

        new_frames.append(name)
        upload_func(0, name, path)

    assert len(new_frames) == num_frames

    return {
        "segments": new_frames,
        "benchmark_bucket": benchmarks_bucket,
        "input_bucket": input_buckets[0],
        "output_bucket": output_buckets[0],
        "batch_size": batch_size,
    }
