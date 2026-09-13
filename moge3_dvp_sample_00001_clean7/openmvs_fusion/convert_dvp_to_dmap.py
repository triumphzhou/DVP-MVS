#!/usr/bin/env python3
"""Convert DVP-MVS depth maps to OpenMVS DMAP files using exact scene metadata."""

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np


HEADER = struct.Struct("<HBBIIIIff")
MAGIC = int.from_bytes(b"DR", "little")
HAS_DEPTH = 1
HAS_NORMAL = 2


def read_dmb(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        channels, height, width, type_code = struct.unpack("<4i", f.read(16))
        if type_code not in (5, 21) or channels not in (1, 3):
            raise ValueError(f"unsupported DMB header in {path}: {(type_code, height, width, channels)}")
        data = np.fromfile(f, dtype="<f4")
    expected = height * width * channels
    if data.size != expected:
        raise ValueError(f"truncated DMB {path}: {data.size} != {expected}")
    return data.reshape(height, width) if channels == 1 else data.reshape(height, width, channels)


def read_template(path: Path):
    with path.open("rb") as f:
        header = HEADER.unpack(f.read(HEADER.size))
        if header[0] != MAGIC:
            raise ValueError(f"bad DMAP magic in {path}")
        name_len = struct.unpack("<H", f.read(2))[0]
        name = f.read(name_len)
        n_ids = struct.unpack("<I", f.read(4))[0]
        ids = np.fromfile(f, dtype="<u4", count=n_ids)
        K = np.fromfile(f, dtype="<f8", count=9).reshape(3, 3)
        R = np.fromfile(f, dtype="<f8", count=9).reshape(3, 3)
        C = np.fromfile(f, dtype="<f8", count=3)
    return header, name, ids, K, R, C


def camera_normals(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Estimate unit camera-space normals, facing the reference camera."""
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xx.astype(np.float32) - cx) / fx
    y = (yy.astype(np.float32) - cy) / fy
    xyz = np.stack((x * depth, y * depth, depth), axis=-1)

    dx = np.empty_like(xyz)
    dy = np.empty_like(xyz)
    dx[:, 1:-1] = xyz[:, 2:] - xyz[:, :-2]
    dx[:, 0] = xyz[:, 1] - xyz[:, 0]
    dx[:, -1] = xyz[:, -1] - xyz[:, -2]
    dy[1:-1] = xyz[2:] - xyz[:-2]
    dy[0] = xyz[1] - xyz[0]
    dy[-1] = xyz[-1] - xyz[-2]

    normal = np.cross(dx, dy)
    length = np.linalg.norm(normal, axis=2)
    valid = depth > 0
    good = valid & np.isfinite(length) & (length > 1e-8)
    normal[good] /= length[good, None]

    ray = np.stack((x, y, np.ones_like(x)), axis=-1)
    ray /= np.linalg.norm(ray, axis=2, keepdims=True)
    normal[valid & ~good] = -ray[valid & ~good]
    flip = valid & (np.sum(normal * ray, axis=2) > 0)
    normal[flip] *= -1
    normal[~valid] = 0
    return np.ascontiguousarray(normal, dtype="<f4")


def write_dmap(
    path: Path,
    template,
    depth: np.ndarray,
    normal: np.ndarray,
    depth_min: float,
    depth_max: float,
):
    header, name, ids, K, R, C = template
    _, _, _, iw, ih, width, height, _, _ = header
    if depth.shape != (height, width):
        raise ValueError(f"shape mismatch for {path.name}: {depth.shape} != {(height, width)}")
    out_header = (MAGIC, HAS_DEPTH | HAS_NORMAL, 0, iw, ih, width, height, depth_min, depth_max)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        f.write(HEADER.pack(*out_header))
        f.write(struct.pack("<H", len(name)))
        f.write(name)
        f.write(struct.pack("<I", len(ids)))
        f.write(np.asarray(ids, dtype="<u4").tobytes())
        f.write(np.asarray(K, dtype="<f8").tobytes())
        f.write(np.asarray(R, dtype="<f8").tobytes())
        f.write(np.asarray(C, dtype="<f8").tobytes())
        f.write(np.ascontiguousarray(depth, dtype="<f4").tobytes())
        f.write(normal.tobytes())
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--templates", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--depth-min", type=float, default=0.5)
    ap.add_argument("--depth-max", type=float, default=80.0)
    args = ap.parse_args()
    if not 0 < args.depth_min < args.depth_max:
        ap.error("expected 0 < --depth-min < --depth-max")

    manifest = json.loads((args.run / "manifest.json").read_text())
    by_label = {item["source_label"]: item for item in manifest["images"]}
    args.output.mkdir(parents=True, exist_ok=True)
    templates = sorted(args.templates.glob("depth*.dmap"))
    if len(templates) != len(by_label):
        raise RuntimeError(f"template/image count mismatch: {len(templates)} != {len(by_label)}")

    counts = []
    seen = set()
    for index, template_path in enumerate(templates, 1):
        template = read_template(template_path)
        label = Path(template[1].decode("utf-8")).stem
        if label not in by_label:
            raise KeyError(f"no DVP image for OpenMVS label {label}")
        item = by_label[label]
        dvp_id = item["id"]
        depth = read_dmb(args.run / "scene" / "APD" / f"{dvp_id:08d}" / "depths.dmb")
        mask = cv2.imread(str(args.run / "scene" / "blocks" / f"mask_{dvp_id}.jpg"), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != depth.shape:
            raise ValueError(f"missing or mismatched block mask for DVP id {dvp_id}")

        px, py = item["padding_xy"]
        cw, ch = item["content_size"]
        depth = depth[py:py + ch, px:px + cw].copy()
        mask = mask[py:py + ch, px:px + cw]
        valid = (
            np.isfinite(depth)
            & (depth >= args.depth_min)
            & (depth <= args.depth_max)
            & (mask >= 128)
        )
        depth[~valid] = 0
        depth = np.ascontiguousarray(depth, dtype="<f4")
        normal = camera_normals(depth, template[3])
        write_dmap(
            args.output / template_path.name,
            template,
            depth,
            normal,
            args.depth_min,
            args.depth_max,
        )
        seen.add(label)
        counts.append(int(valid.sum()))
        if index % 50 == 0 or index == len(templates):
            print(f"converted {index}/{len(templates)}; last={label}; valid={counts[-1]}", flush=True)

    if seen != set(by_label):
        raise RuntimeError(f"unmatched labels: {sorted(set(by_label) - seen)[:10]}")
    stats = {
        "files": len(counts),
        "valid_depth_pixels": int(sum(counts)),
        "valid_per_image_min": int(min(counts)),
        "valid_per_image_median": float(np.median(counts)),
        "valid_per_image_max": int(max(counts)),
        "depth_range_m": [args.depth_min, args.depth_max],
        "dmap_channels": ["depth", "camera_normal"],
    }
    (args.output.parent / "conversion_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
