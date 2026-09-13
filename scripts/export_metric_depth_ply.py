#!/usr/bin/env python3
"""Export saved metric-prior DMB depth maps as world-coordinate PLY files."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np


def read_dmb(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        channels, height, width, type_code = struct.unpack("<4i", stream.read(16))
        if channels != 1 or type_code != 5:
            raise ValueError(f"Unsupported DMB header in {path}: {(channels, height, width, type_code)}")
        depth = np.fromfile(stream, dtype="<f4")
    if depth.size != height * width:
        raise ValueError(f"Truncated DMB file: {path}")
    return depth.reshape(height, width)


def read_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    extrinsic_at = lines.index("extrinsic")
    intrinsic_at = lines.index("intrinsic")
    world_to_camera = np.array(
        [[float(value) for value in lines[extrinsic_at + 1 + row].split()] for row in range(4)],
        dtype=np.float64,
    )
    intrinsic = np.array(
        [[float(value) for value in lines[intrinsic_at + 1 + row].split()] for row in range(3)],
        dtype=np.float64,
    )
    return world_to_camera, intrinsic


def make_points(
    depth: np.ndarray,
    bgr: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    minimum: float,
    maximum: float,
    depth_colors: bool,
) -> np.ndarray:
    valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    y, x = np.nonzero(valid)
    z = depth[y, x].astype(np.float64)
    xyz_camera = np.column_stack(
        ((x - intrinsic[0, 2]) * z / intrinsic[0, 0],
         (y - intrinsic[1, 2]) * z / intrinsic[1, 1],
         z)
    )
    camera_to_world = np.linalg.inv(world_to_camera)
    xyz_world = xyz_camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    ranges = np.linalg.norm(xyz_camera, axis=1)

    if depth_colors:
        normalized = np.clip((z - minimum) / max(maximum - minimum, 1e-6), 0, 1)
        color_bgr = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        rgb = color_bgr[:, 0, ::-1]
    else:
        rgb = bgr[y, x, ::-1]

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("depth_z", "<f4"), ("range", "<f4"),
    ])
    points = np.empty(len(z), dtype=dtype)
    points["x"], points["y"], points["z"] = xyz_world.T.astype(np.float32)
    points["red"], points["green"], points["blue"] = rgb.T
    points["depth_z"] = z.astype(np.float32)
    points["range"] = ranges.astype(np.float32)
    return points


def write_ply(path: Path, points: np.ndarray) -> None:
    header = "\n".join([
        "ply", "format binary_little_endian 1.0",
        f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "property float depth_z", "property float range", "end_header", "",
    ]).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        points.tofile(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-min", type=float, default=0.5)
    parser.add_argument("--depth-max", type=float, default=80.0)
    parser.add_argument("--far-min", type=float, default=40.0)
    args = parser.parse_args()

    manifest = json.loads((args.run / "manifest.json").read_text())
    by_label = {item["source_label"]: item for item in manifest["images"]}
    args.output.mkdir(parents=True, exist_ok=True)
    combined = []
    report = {"depth_convention": "camera optical-axis Z in meters", "items": []}

    for label in args.labels:
        item = by_label[label]
        image_id = int(item["id"])
        stem = f"{image_id:08d}"
        depth = read_dmb(args.run / "scene" / "metric_prior" / f"{stem}.dmb")
        bgr = cv2.imread(str(args.run / "scene" / "images" / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != depth.shape:
            raise ValueError(f"Missing or mismatched image for {label}")
        world_to_camera, intrinsic = read_camera(args.run / "scene" / "cams" / f"{stem}_cam.txt")

        full = make_points(depth, bgr, world_to_camera, intrinsic, args.depth_min, args.depth_max, False)
        far = make_points(depth, bgr, world_to_camera, intrinsic, args.far_min, args.depth_max, True)
        write_ply(args.output / f"{label}_moge_metric_full.ply", full)
        write_ply(args.output / f"{label}_moge_metric_far_{args.far_min:g}_{args.depth_max:g}m.ply", far)
        combined.append(far)
        report["items"].append({
            "label": label, "image_id": image_id, "full_points": len(full), "far_points": len(far),
            "far_fraction": len(far) / max(len(full), 1),
            "depth_z_min": float(full["depth_z"].min()), "depth_z_max": float(full["depth_z"].max()),
            "euclidean_range_max": float(full["range"].max()),
        })

    merged = np.concatenate(combined)
    write_ply(args.output / f"combined_moge_metric_far_{args.far_min:g}_{args.depth_max:g}m.ply", merged)
    report["combined_far_points"] = len(merged)
    (args.output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
