#!/usr/bin/env python3
"""Validate the completed MoGe-3 + DVP-MVS all-image run."""

import hashlib
import json
import math
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scene"
VIEW_COUNT = 707
WIDTH, HEIGHT = 960, 640


def dmb_header(path: Path):
    with path.open("rb") as stream:
        return struct.unpack("4i", stream.read(16))


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    scales = [item["moge_scale_multiplier"] for item in manifest["images"]]
    coverage = [item["valid_prior_fraction"] for item in manifest["images"]]
    lidar_counts = [item["lidar_calibration_pixels"] for item in manifest["images"]]

    input_depth = sorted((SCENE / "metric_prior").glob("[0-9]*.dmb"))
    input_depth = [path for path in input_depth if not path.name.endswith("_normal.dmb")]
    input_normal = sorted((SCENE / "metric_prior").glob("*_normal.dmb"))
    output_depth = sorted((SCENE / "APD").glob("*/depths.dmb"))
    output_normal = sorted((SCENE / "APD").glob("*/APD_normals.dmb"))
    assert len(input_depth) == len(input_normal) == len(output_depth) == len(output_normal) == VIEW_COUNT
    for path in input_depth + output_depth:
        assert dmb_header(path) == (1, HEIGHT, WIDTH, 5), path
        assert path.stat().st_size == 16 + WIDTH * HEIGHT * 4, path
    for path in input_normal + output_normal:
        assert dmb_header(path) == (1, HEIGHT, WIDTH, 21), path
        assert path.stat().st_size == 16 + WIDTH * HEIGHT * 3 * 4, path

    ply = SCENE / "APD" / "APD.ply"
    header = b""
    with ply.open("rb") as stream:
        while not header.endswith(b"end_header\n"):
            header += stream.readline()
    vertex_match = re.search(rb"element vertex (\d+)", header)
    assert vertex_match
    vertices = int(vertex_match.group(1))

    log = (ROOT / "dvp.log").read_text(errors="replace")
    error_patterns = re.findall(
        r"cuda.*error|invalid.*(?:read|write)|failure|segmentation|terminate called",
        log,
        flags=re.IGNORECASE,
    )
    assert log.rstrip().endswith("All done")
    assert not error_patterns

    digest = hashlib.sha256()
    with ply.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    report = {
        "status": "passed",
        "input_images": VIEW_COUNT,
        "cameras": manifest["cameras"],
        "output_resolution": [WIDTH, HEIGHT],
        "moge_metric_prior_depth_maps": len(input_depth),
        "moge_metric_prior_normal_maps": len(input_normal),
        "dvp_depth_maps": len(output_depth),
        "dvp_normal_maps": len(output_normal),
        "moge_scale_min_mean_max": [min(scales), sum(scales) / len(scales), max(scales)],
        "moge_valid_fraction_min_mean_max": [min(coverage), sum(coverage) / len(coverage), max(coverage)],
        "lidar_calibration_pixels_min_mean_max": [min(lidar_counts), sum(lidar_counts) / len(lidar_counts), max(lidar_counts)],
        "point_cloud_vertices": vertices,
        "point_cloud_bytes": ply.stat().st_size,
        "point_cloud_sha256": digest.hexdigest(),
        "dvp_log_ends_all_done": True,
        "dvp_error_pattern_count": 0,
        "apd_is_physical_workspace_directory": not (SCENE / "APD").is_symlink(),
    }
    (ROOT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
