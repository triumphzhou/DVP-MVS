#!/usr/bin/env python3
"""Validate all generated priors and create DVP-MVS pair.txt/manifest.json."""

import json
import struct
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scene"
SOURCE = Path(
    "/mnt/nuplan/l3data-reconstruction-bingxing/samples/"
    "sample_00001_clip_M18-2_07_20251202093910_DF_f76_176_left/converted"
)
WIDTH, HEIGHT = 960, 640


def check_dmb(path: Path, channels: int) -> None:
    expected_size = 16 + WIDTH * HEIGHT * channels * 4
    if not path.is_file() or path.stat().st_size != expected_size:
        raise RuntimeError(f"Invalid DMB: {path} (expected {expected_size} bytes)")
    version, rows, cols, cv_type = struct.unpack("4i", path.read_bytes()[:16])
    expected_type = 5 if channels == 1 else 21
    if (version, rows, cols, cv_type) != (1, HEIGHT, WIDTH, expected_type):
        raise RuntimeError(f"Invalid DMB header: {path}")


def main() -> None:
    source_images = sorted((SOURCE / "images").glob("*.jpg"))
    records = []
    by_camera = defaultdict(list)
    for image_id, source_image in enumerate(source_images):
        stem = f"{image_id:08d}"
        check_dmb(SCENE / "metric_prior" / f"{stem}.dmb", 1)
        check_dmb(SCENE / "metric_prior" / f"{stem}_normal.dmb", 3)
        for required in (
            SCENE / "images" / f"{stem}.jpg",
            SCENE / "cams" / f"{stem}_cam.txt",
            SCENE / "metadata" / f"{stem}.json",
        ):
            if not required.is_file():
                raise RuntimeError(f"Missing {required}")
        record = json.loads((SCENE / "metadata" / f"{stem}.json").read_text())
        if record["source_label"] != source_image.stem:
            raise RuntimeError(f"ID mapping mismatch for {stem}")
        records.append(record)
        frame_text, camera = source_image.stem.split("_")
        by_camera[camera].append((int(frame_text), image_id))

    pair_lines = [str(len(records))]
    source_counts = []
    for record in records:
        frame_text, camera = record["source_label"].split("_")
        frame = int(frame_text)
        candidates = sorted(
            ((abs(other_frame - frame), other_id) for other_frame, other_id in by_camera[camera] if other_id != record["id"]),
            key=lambda value: (value[0], value[1]),
        )[:6]
        if len(candidates) < 2:
            raise RuntimeError(f"Too few source views for {record['source_label']}")
        pair_lines.append(str(record["id"]))
        pair_lines.append(
            str(len(candidates))
            + " "
            + " ".join(f"{source_id} {1.0 / (1.0 + distance):.6f}" for distance, source_id in candidates)
        )
        record["source_views"] = [source_id for _, source_id in candidates]
        source_counts.append(len(candidates))

    (SCENE / "pair.txt").write_text("\n".join(pair_lines) + "\n")
    manifest = {
        "source": str(SOURCE),
        "output_size": [WIDTH, HEIGHT],
        "image_count": len(records),
        "camera_count": len(by_camera),
        "cameras": sorted(by_camera),
        "prior": "MoGe-3 depth calibrated per view by median sparse LiDAR ratio; normals derived from calibrated depth and camera intrinsics",
        "pairing": "six nearest temporal frames from the same camera",
        "source_views_min_max": [min(source_counts), max(source_counts)],
        "images": records,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
