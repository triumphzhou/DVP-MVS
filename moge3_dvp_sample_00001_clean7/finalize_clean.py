#!/usr/bin/env python3
"""Robustly calibrate priors, filter them, and build the verified 20-view graph."""

import csv
import json
import math
import os
import struct
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(os.environ.get("DVP_RUN_ROOT", Path(__file__).resolve().parent))
SCENE = ROOT / "scene"
SOURCE = Path(os.environ.get(
    "DVP_SOURCE",
    "/mnt/nuplan/l3data-reconstruction-bingxing/samples/"
    "sample_00001_clip_M18-2_07_20251202093910_DF_f76_176_left/converted",
))
NEIGHBORS_TSV = Path(os.environ.get(
    "DVP_NEIGHBORS_TSV",
    "/mnt/nuplan/l3data-reconstruction-bingxing/tem-test/colmap+openmvs/"
    "batch8_first8_roadmesh/results/"
    "sample_00001_clip_M18-2_07_20251202093910_DF_f76_176_left/"
    "00_audit/neighbors_diverse_pm20_top20.tsv",
))
WIDTH, HEIGHT = 960, 640
DEPTH_MIN, DEPTH_MAX = 0.5, 80.0
MIN_PRIOR_COMPONENT = 20
FINALIZATION_VERSION = "clean7-v1"


def atomic_bytes(path: Path, data: bytes) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def read_dmb(path: Path, channels: int) -> np.ndarray:
    with path.open("rb") as f:
        header = f.read(16)
        version, rows, cols, cv_type = struct.unpack("4i", header)
        expected_type = 5 if channels == 1 else 21
        if (version, rows, cols, cv_type) != (1, HEIGHT, WIDTH, expected_type):
            raise RuntimeError(f"Invalid DMB header: {path}")
        data = np.frombuffer(f.read(), dtype=np.float32)
    expected = HEIGHT * WIDTH * channels
    if data.size != expected:
        raise RuntimeError(f"Invalid DMB payload: {path}")
    shape = (HEIGHT, WIDTH) if channels == 1 else (HEIGHT, WIDTH, channels)
    return data.reshape(shape).copy()


def write_dmb(path: Path, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array, dtype=np.float32)
    cv_type = 5 if array.ndim == 2 else 21
    atomic_bytes(path, struct.pack("4i", 1, array.shape[0], array.shape[1], cv_type) + array.tobytes())


def remove_small_components(depth: np.ndarray) -> int:
    valid = (depth > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(valid, connectivity=8)
    small = np.zeros(count, dtype=bool)
    if count > 1:
        small[1:] = stats[1:, cv2.CC_STAT_AREA] < MIN_PRIOR_COMPONENT
    remove = small[labels]
    removed = int(remove.sum())
    depth[remove] = 0
    return removed


def make_normals(depth: np.ndarray, intrinsic: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.float32)
    xyz = np.stack(
        (
            (xx - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (yy - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
        ),
        axis=-1,
    )
    normal_camera = np.cross(np.gradient(xyz, axis=1), np.gradient(xyz, axis=0))
    normal_camera /= np.maximum(np.linalg.norm(normal_camera, axis=-1, keepdims=True), 1e-8)
    normal_camera[np.sum(normal_camera * xyz, axis=-1) > 0] *= -1
    neighbors_valid = depth > 0
    neighbors_valid[1:-1, 1:-1] &= (
        (depth[:-2, 1:-1] > 0)
        & (depth[2:, 1:-1] > 0)
        & (depth[1:-1, :-2] > 0)
        & (depth[1:-1, 2:] > 0)
    )
    neighbors_valid[[0, -1], :] = False
    neighbors_valid[:, [0, -1]] = False
    normal_world = normal_camera @ world_to_camera[:3, :3]
    normal_world[~neighbors_valid] = 0
    return normal_world.astype(np.float32)


def main() -> None:
    source_images = sorted((SOURCE / "images").glob("*.jpg"))
    if len(source_images) != 707:
        raise RuntimeError(f"Expected 707 source images, got {len(source_images)}")
    records = []
    by_camera = defaultdict(list)
    for image_id, source_image in enumerate(source_images):
        stem = f"{image_id:08d}"
        metadata_path = SCENE / "metadata" / f"{stem}.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing {metadata_path}")
        record = json.loads(metadata_path.read_text())
        if record["source_label"] != source_image.stem:
            raise RuntimeError(f"ID mapping mismatch for {stem}")
        records.append(record)
        camera = source_image.stem.rsplit("_", 1)[1]
        by_camera[camera].append(record)

    calibration_report = {}
    for camera, camera_records in sorted(by_camera.items()):
        logs = np.log([r["raw_moge_scale_multiplier"] for r in camera_records])
        center = float(np.median(logs))
        mad = float(np.median(np.abs(logs - center)))
        robust_sigma = 1.4826 * mad
        limit = min(math.log(1.15), max(math.log(1.10), 3.0 * robust_sigma))
        clipped = 0
        for record, raw_log in zip(camera_records, logs):
            used_log = float(np.clip(raw_log, center - limit, center + limit))
            record["moge_scale_multiplier"] = math.exp(used_log)
            record["scale_was_clipped"] = bool(abs(used_log - raw_log) > 1e-8)
            clipped += int(record["scale_was_clipped"])
        calibration_report[camera] = {
            "count": len(camera_records),
            "median_scale": math.exp(center),
            "log_mad": mad,
            "allowed_ratio": math.exp(limit),
            "clipped_images": clipped,
        }

    total_prior_speckles = 0
    for record in records:
        image_id = record["id"]
        stem = f"{image_id:08d}"
        raw_scale = record["raw_moge_scale_multiplier"]
        used_scale = record["moge_scale_multiplier"]
        depth_path = SCENE / "metric_prior" / f"{stem}.dmb"
        if record.get("finalization_version") == FINALIZATION_VERSION:
            total_prior_speckles += int(record.get("prior_small_component_pixels_removed", 0))
            continue
        depth = read_dmb(depth_path, 1)
        depth *= used_scale / raw_scale
        block = cv2.imread(str(SCENE / "blocks" / f"mask_{image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
        if block is None or block.shape != (HEIGHT, WIDTH):
            raise RuntimeError(f"Invalid DVP block mask for {stem}")
        depth[(block < 128) | ~np.isfinite(depth) | (depth < DEPTH_MIN) | (depth > DEPTH_MAX)] = 0
        removed = remove_small_components(depth)
        total_prior_speckles += removed

        label = record["source_label"]
        source_width, source_height = record["source_size"]
        resize_scale = record["resize_scale"]
        pad_x, pad_y = record["padding_xy"]
        intrinsic = np.loadtxt(SOURCE / "intrinsics" / f"{label}.txt", dtype=np.float64)
        intrinsic[0, :] *= resize_scale
        intrinsic[1, :] *= resize_scale
        intrinsic[0, 2] += pad_x
        intrinsic[1, 2] += pad_y
        camera_to_world = np.loadtxt(SOURCE / "ego_pose" / f"{label}.txt", dtype=np.float64)
        world_to_camera = np.linalg.inv(camera_to_world)
        normals = make_normals(depth, intrinsic, world_to_camera)
        write_dmb(depth_path, depth)
        write_dmb(SCENE / "metric_prior" / f"{stem}_normal.dmb", normals)
        record["valid_prior_fraction"] = float(np.mean(depth > 0))
        record["prior_small_component_pixels_removed"] = removed
        record["finalization_version"] = FINALIZATION_VERSION
        atomic_bytes(SCENE / "metadata" / f"{stem}.json", (json.dumps(record, indent=2) + "\n").encode())

    label_to_id = {record["source_label"]: record["id"] for record in records}
    selected = defaultdict(list)
    with NEIGHBORS_TSV.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ref = row["reference"]
            src = row["source"]
            selected[ref].append((int(row["rank"]), src, int(row["shared_sparse_points"])))

    pair_lines = [str(len(records))]
    cross_counts = []
    for record in records:
        label = record["source_label"]
        neighbors = sorted(selected[label])
        if len(neighbors) != 20:
            raise RuntimeError(f"Expected 20 verified neighbors for {label}, got {len(neighbors)}")
        sources = [(label_to_id[src], evidence) for _, src, evidence in neighbors]
        pair_lines.append(str(record["id"]))
        pair_lines.append("20 " + " ".join(f"{source_id} {max(evidence, 1):.6f}" for source_id, evidence in sources))
        record["source_views"] = [source_id for source_id, _ in sources]
        ref_camera = label.rsplit("_", 1)[1]
        cross = sum(records[source_id]["source_label"].rsplit("_", 1)[1] != ref_camera for source_id, _ in sources)
        cross_counts.append(cross)

    atomic_bytes(SCENE / "pair.txt", ("\n".join(pair_lines) + "\n").encode())
    manifest = {
        "source": str(SOURCE),
        "output_size": [WIDTH, HEIGHT],
        "image_count": len(records),
        "camera_count": len(by_camera),
        "cameras": sorted(by_camera),
        "changes": {
            "masks": "OpenMVS conservative sky masks for every camera plus camera10 fixed ego-rig mask; applied to priors and DVP fusion",
            "pairing": "20 COLMAP-SIFT-verified OpenMVS views per reference, including cross-camera and +/-20-frame diversity",
            "fusion_min_consistent_source_views": 2,
            "resize": "aspect-preserving fit into 960x640; camera09/10 are 960x540 with 50-pixel top/bottom padding",
            "scale_filter": "per-camera log-scale median/MAD clipping, constrained to 10--15 percent",
            "prior_component_filter_pixels": MIN_PRIOR_COMPONENT,
            "fusion_depth_component_filter_pixels": 7,
            "fusion_depth_component_relative_threshold": 0.05,
            "depth_range_m": [DEPTH_MIN, DEPTH_MAX],
        },
        "calibration": calibration_report,
        "prior_small_component_pixels_removed": total_prior_speckles,
        "source_views_min_max": [20, 20],
        "cross_camera_sources_min_max": [min(cross_counts), max(cross_counts)],
        "images": records,
    }
    atomic_bytes(ROOT / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({key: value for key, value in manifest.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
