"""
Log one or more PCD files to rerun.io.

Usage (with uv):
  uv run python log_pcds.py path/to/a.pcd [path/to/b.pcd ...]

Notes:
- Colors are used if present in the PCD; otherwise a distinct color is assigned per file.
- Requires Python >= 3.9 for rerun. If you're on 3.8, run with:
    uv run --python 3.11 python log_pcds.py <pcd...>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import open3d as o3d
from tqdm import tqdm


def load_pcd(path: str) -> o3d.geometry.PointCloud:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PCD not found: {path}")
    pc = o3d.io.read_point_cloud(str(p))
    if pc.is_empty():
        raise ValueError(f"Point cloud is empty: {path}")
    return pc


def to_numpy(pc: o3d.geometry.PointCloud) -> tuple[np.ndarray, Optional[np.ndarray]]:
    pts = np.asarray(pc.points)
    cols = None
    if pc.has_colors():
        cols = (np.asarray(pc.colors) * 255.0).astype(np.uint8)
        if cols.ndim == 2 and cols.shape[1] == 3:
            alpha = np.full((cols.shape[0], 1), 255, dtype=np.uint8)
            cols = np.concatenate([cols, alpha], axis=1)
    return pts, cols


def default_color(idx: int) -> np.ndarray:
    palette = [
        (255, 99, 99, 255),   # red-ish
        (99, 181, 255, 255),  # blue-ish
        (99, 255, 148, 255),  # green-ish
        (255, 214, 99, 255),  # orange-ish
        (207, 99, 255, 255),  # purple-ish
        (99, 255, 240, 255),  # cyan-ish
    ]
    return np.array(palette[idx % len(palette)], dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log one or more PCDs to rerun.io")
    parser.add_argument("pcds", nargs="+", help="Paths to .pcd files")
    args = parser.parse_args()

    try:
        import rerun as rr
    except Exception as e:
        raise SystemExit(
            "Failed to import rerun. This often happens on Python 3.8 due to typing features.\n"
            "Try running with a newer Python using uv:\n"
            "  uv run --python 3.11 python log_pcds.py <pcd...>\n\n"
            f"Original import error: {e}"
        )

    import rerun as rr
    rr.init("PCD Viewer")
    rr.spawn()

    for i, pcd_path in enumerate(tqdm(args.pcds, desc="Processing PCD files")):
        name = Path(pcd_path).stem
        try:
            pc = load_pcd(pcd_path)
        except Exception as e:
            print(f"Skipping '{pcd_path}': {e}")
            continue

        pts, cols = to_numpy(pc)
        if cols is None:
            col = default_color(i)
            cols = np.repeat(col[None, :], pts.shape[0], axis=0)

        # Create voxel grid version for visualization (better performance)
        voxel_size = 0.01  # 1cm voxels for good balance of detail and performance
        pc_downsampled = pc.voxel_down_sample(voxel_size=voxel_size)
        pts_downsampled = np.asarray(pc_downsampled.points)
        
        # Update colors for downsampled version
        if pc_downsampled.has_colors():
            cols_downsampled = (np.asarray(pc_downsampled.colors) * 255.0).astype(np.uint8)
            if cols_downsampled.ndim == 2 and cols_downsampled.shape[1] == 3:
                alpha = np.full((cols_downsampled.shape[0], 1), 255, dtype=np.uint8)
                cols_downsampled = np.concatenate([cols_downsampled, alpha], axis=1)
        else:
            cols_downsampled = cols[:len(pts_downsampled)]  # Use original colors for downsampled points

        print(f"Visualizing {name}: {len(pts_downsampled)} points (downsampled from {len(pts)} for performance)")
        rr.log(f"cloud/{name}", rr.Points3D(positions=pts_downsampled, colors=cols_downsampled, radii=0.003))

        # Log bounding box
        bbx = pc.get_axis_aligned_bounding_box()
        center = bbx.get_center()
        half_size = bbx.get_extent() / 2
        rr.log(f"cloud/{name}_bbox", rr.Boxes3D(
            centers=np.array([center]),
            half_sizes=np.array([half_size])
        ))


if __name__ == "__main__":
    main()
