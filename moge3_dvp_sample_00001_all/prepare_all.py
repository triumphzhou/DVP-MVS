#!/usr/bin/env python3
"""Run MoGe-3 for one GPU shard and prepare metric priors for DVP-MVS."""

import argparse
import json
import os
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    "/mnt/nuplan/l3data-reconstruction-bingxing/samples/"
    "sample_00001_clip_M18-2_07_20251202093910_DF_f76_176_left/converted"
)
SCENE = ROOT / "scene"
WEIGHTS = Path("/mnt/zhoukaixuan_workspace/code/weights/moge-3-vitl/model.pt")
WIDTH, HEIGHT = 960, 640


def atomic_bytes(path: Path, data: bytes) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def write_dmb(path: Path, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array, dtype=np.float32)
    cv_type = 5 if array.ndim == 2 else 21
    payload = struct.pack("4i", 1, array.shape[0], array.shape[1], cv_type)
    atomic_bytes(path, payload + array.tobytes())


def dmb_valid(path: Path, channels: int) -> bool:
    return path.is_file() and path.stat().st_size == 16 + WIDTH * HEIGHT * channels * 4


def write_camera(path: Path, extrinsic: np.ndarray, intrinsic: np.ndarray) -> None:
    text = "extrinsic\n"
    text += "\n".join(" ".join(f"{v:.12g}" for v in row) for row in extrinsic)
    text += "\n\nintrinsic\n"
    text += "\n".join(" ".join(f"{v:.12g}" for v in row) for row in intrinsic)
    text += "\n\n0.8333333333 0.25 397 100\n"
    atomic_bytes(path, text.encode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--refine-steps", type=int, default=3)
    args = parser.parse_args()

    for name in ("images", "cams", "metric_prior", "metadata"):
        (SCENE / name).mkdir(parents=True, exist_ok=True)

    image_paths = sorted((SOURCE / "images").glob("*.jpg"))
    selected = [(i, p) for i, p in enumerate(image_paths) if i % args.num_shards == args.shard]
    pending = []
    for image_id, image_path in selected:
        stem = f"{image_id:08d}"
        if (
            dmb_valid(SCENE / "metric_prior" / f"{stem}.dmb", 1)
            and dmb_valid(SCENE / "metric_prior" / f"{stem}_normal.dmb", 3)
            and (SCENE / "images" / f"{stem}.jpg").is_file()
            and (SCENE / "cams" / f"{stem}_cam.txt").is_file()
            and (SCENE / "metadata" / f"{stem}.json").is_file()
        ):
            print(f"[{args.shard}] skip complete {stem} {image_path.stem}", flush=True)
        else:
            pending.append((image_id, image_path))

    if not pending:
        print(f"[{args.shard}] all {len(selected)} images already complete", flush=True)
        return

    from moge.model import import_model_class_by_version

    model = import_model_class_by_version("v3").from_pretrained(str(WEIGHTS)).cuda().eval()
    started = time.time()
    for done, (image_id, image_path) in enumerate(pending, 1):
        label = image_path.stem
        stem = f"{image_id:08d}"
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read {image_path}")
        source_height, source_width = bgr.shape[:2]
        resized = cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float().div_(255.0)

        output = model.infer(
            tensor,
            resolution_level=args.resolution_level,
            refine_steps=args.refine_steps,
            use_fp16=True,
        )
        predicted = output["depth"].detach().cpu().numpy().astype(np.float32)
        mask = output["mask"].detach().cpu().numpy().astype(bool)
        del tensor, output

        lidar = np.load(SOURCE / "lidar_depth" / f"{label}.npy", allow_pickle=True).item()
        source_y, source_x = np.nonzero(lidar["mask"])
        lidar_values = np.asarray(lidar["value"], dtype=np.float32)
        resized_x = np.clip(np.rint((source_x + 0.5) * WIDTH / source_width - 0.5).astype(int), 0, WIDTH - 1)
        resized_y = np.clip(np.rint((source_y + 0.5) * HEIGHT / source_height - 0.5).astype(int), 0, HEIGHT - 1)
        sampled = predicted[resized_y, resized_x]
        calibration = (
            np.isfinite(sampled)
            & (sampled > 0)
            & mask[resized_y, resized_x]
            & np.isfinite(lidar_values)
            & (lidar_values > 2.0)
            & (lidar_values < 80.0)
        )
        ratios = lidar_values[calibration] / sampled[calibration]
        if ratios.size < 100:
            raise RuntimeError(f"Only {ratios.size} LiDAR calibration pixels for {label}")
        scale = float(np.median(ratios))
        depth = np.where(mask & np.isfinite(predicted) & (predicted > 0), predicted * scale, 0.0)
        depth[(depth < 0.5) | (depth > 120.0)] = 0.0
        depth = depth.astype(np.float32)

        intrinsic = np.loadtxt(SOURCE / "intrinsics" / f"{label}.txt", dtype=np.float64)
        camera_to_world = np.loadtxt(SOURCE / "ego_pose" / f"{label}.txt", dtype=np.float64)
        world_to_camera = np.linalg.inv(camera_to_world)
        intrinsic[0, :] *= WIDTH / source_width
        intrinsic[1, :] *= HEIGHT / source_height

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
        normal_world = normal_camera @ world_to_camera[:3, :3]
        normal_world[depth == 0] = 0

        write_dmb(SCENE / "metric_prior" / f"{stem}.dmb", depth)
        write_dmb(SCENE / "metric_prior" / f"{stem}_normal.dmb", normal_world)
        image_temp = SCENE / "images" / f"{stem}.tmp.jpg"
        if not cv2.imwrite(str(image_temp), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Cannot write {image_temp}")
        os.replace(image_temp, SCENE / "images" / f"{stem}.jpg")
        write_camera(SCENE / "cams" / f"{stem}_cam.txt", world_to_camera, intrinsic)

        metadata = {
            "id": image_id,
            "source_label": label,
            "source_size": [source_width, source_height],
            "moge_scale_multiplier": scale,
            "lidar_calibration_pixels": int(ratios.size),
            "valid_prior_fraction": float(np.mean(depth > 0)),
        }
        atomic_bytes(
            SCENE / "metadata" / f"{stem}.json",
            (json.dumps(metadata, indent=2) + "\n").encode(),
        )
        elapsed = time.time() - started
        print(
            f"[{args.shard}] {done}/{len(pending)} id={stem} label={label} "
            f"scale={scale:.5f} valid={metadata['valid_prior_fraction']:.3f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
