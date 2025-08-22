"""
Load a PCD with Open3D, cut a square from the center of mass with a height cap, voxelize the result, and show it in rerun.io.

Usage examples (using uv):
  - Default (auto-pick a PCD, crop 50% square on XY, voxel=0.01):
	  uv run python prepare_cloud.py

  - Explicit file and square side length (units = point units):
	  uv run python prepare_cloud.py --pcd dataset/pcds/02691156/1e40d41905a9be766ed8c57a1980bb26.pcd --side 0.3

	- Choose plane, voxel size, and height multiplier (height = multiplier * side along the axis normal to the plane):
			uv run python prepare_cloud.py --plane yz --ratio 0.4 --voxel 0.02 --height-mult 2.0

  - Headless check (no viewer; prints counts; still saves unless --no-save):
	  uv run python prepare_cloud.py --dry-run
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
# Note: rerun is imported lazily inside visualize_rerun to allow --dry-run on older Python versions.


def _auto_find_pcd() -> str | None:
	"""Try to find a PCD under dataset/pcds/**. Return first match or None."""
	candidates = sorted(
		glob.glob("dataset/pcds/**/*.pcd", recursive=True)
	)
	return candidates[0] if candidates else None


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"PCD not found: {path}")
	pc = o3d.io.read_point_cloud(str(p))
	if pc.is_empty():
		raise ValueError(f"Point cloud is empty: {path}")
	return pc


def compute_centroid(points: np.ndarray) -> np.ndarray:
	if points.ndim != 2 or points.shape[1] != 3:
		raise ValueError("points must be of shape (N, 3)")
	return points.mean(axis=0)


def infer_side_length(points: np.ndarray, plane: str, ratio: float) -> float:
	# Axis indices for plane
	ax = {
		"xy": (0, 1),
		"yz": (1, 2),
		"xz": (0, 2),
	}[plane]
	mins = points[:, list(ax)].min(axis=0)
	maxs = points[:, list(ax)].max(axis=0)
	ranges = maxs - mins
	# Use a fraction of the smaller span to get a reasonably sized square
	return float(min(ranges) * ratio)


def crop_square_mask(points: np.ndarray, center: np.ndarray, plane: str, side: float, height_mult: float = 2.0) -> np.ndarray:
	"""Create a mask for a square on the given plane with a capped height along the orthogonal axis.

	- plane=xy -> square in X/Y, capped in Z by height = height_mult * side
	- plane=yz -> square in Y/Z, capped in X by height = height_mult * side
	- plane=xz -> square in X/Z, capped in Y by height = height_mult * side
	"""
	if side <= 0:
		raise ValueError("side must be positive")
	if height_mult <= 0:
		raise ValueError("height_mult must be positive")
	half = side / 2.0
	hhalf = (height_mult * side) / 2.0
	if plane == "xy":
		m = (
			(np.abs(points[:, 0] - center[0]) <= half)
			& (np.abs(points[:, 1] - center[1]) <= half)
			& (np.abs(points[:, 2] - center[2]) <= hhalf)
		)
	elif plane == "yz":
		m = (
			(np.abs(points[:, 1] - center[1]) <= half)
			& (np.abs(points[:, 2] - center[2]) <= half)
			& (np.abs(points[:, 0] - center[0]) <= hhalf)
		)
	elif plane == "xz":
		m = (
			(np.abs(points[:, 0] - center[0]) <= half)
			& (np.abs(points[:, 2] - center[2]) <= half)
			& (np.abs(points[:, 1] - center[1]) <= hhalf)
		)
	else:
		raise ValueError("plane must be one of 'xy', 'yz', 'xz'")
	return m


def to_numpy(pc: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray | None]:
	pts = np.asarray(pc.points)
	cols = None
	if pc.has_colors():
		cols = (np.asarray(pc.colors) * 255.0).astype(np.uint8)
		if cols.shape[1] == 3:  # add alpha
			alpha = np.full((cols.shape[0], 1), 255, dtype=np.uint8)
			cols = np.concatenate([cols, alpha], axis=1)
	return pts, cols


def visualize_rerun(crop_pts: np.ndarray,
					crop_cols: np.ndarray | None = None,
					title: str = "SC3K - Voxel Crop") -> None:
	try:
		import rerun as rr  # lazy import
	except Exception as e:
		raise SystemExit(
			"Failed to import rerun. This often happens on Python 3.8 due to typing features.\n"
			"Try running with a newer Python using uv:\n"
			"  uv run --python 3.11 python prepare_cloud.py\n\n"
			f"Original import error: {e}"
		)
	rr.init(title)
	rr.spawn()
	# Cropped: highlight color if no colors
	if crop_cols is None:
		cr_color = np.array([255, 90, 90, 255], dtype=np.uint8)
		cr_colors = np.repeat(cr_color[None, :], crop_pts.shape[0], axis=0)
	else:
		cr_colors = crop_cols

	rr.log("cloud/crop_voxel", rr.Points3D(positions=crop_pts, colors=cr_colors, radii=0.006))


def voxel_downsample(points: np.ndarray, colors: np.ndarray | None, voxel: float) -> Tuple[np.ndarray, np.ndarray | None]:
	if voxel <= 0:
		return points, colors
	pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
	if colors is not None:
		# colors expected as RGBA; Open3D expects [0,1] RGB. We'll keep RGB if present and drop alpha for downsample.
		if colors.shape[1] >= 3:
			rgb = (colors[:, :3].astype(np.float32) / 255.0)
			pc.colors = o3d.utility.Vector3dVector(rgb)
	pc_ds = pc.voxel_down_sample(voxel)
	pts_ds = np.asarray(pc_ds.points)
	cols_ds = None
	if pc_ds.has_colors():
		rgb_ds = np.asarray(pc_ds.colors)
		cols_ds = (np.clip(rgb_ds, 0.0, 1.0) * 255.0).astype(np.uint8)
		# add opaque alpha
		if cols_ds.ndim == 2 and cols_ds.shape[1] == 3:
			alpha = np.full((cols_ds.shape[0], 1), 255, dtype=np.uint8)
			cols_ds = np.concatenate([cols_ds, alpha], axis=1)
	return pts_ds, cols_ds


def _to_o3d_cloud(points: np.ndarray, colors: np.ndarray | None) -> o3d.geometry.PointCloud:
	pc = o3d.geometry.PointCloud()
	pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
	if colors is not None and colors.size > 0:
		# Accept RGBA or RGB uint8 and map to Open3D RGB float [0,1]
		rgb = colors[:, :3].astype(np.float32) / 255.0
		pc.colors = o3d.utility.Vector3dVector(rgb)
	return pc


def _sanitize_float(x: float) -> str:
	s = f"{x:.3f}"
	s = s.rstrip("0").rstrip(".")
	return s if s else "0"


def save_clouds(out_dir: Path,
				in_stem: str,
				plane: str,
				side: float,
				height_mult: float,
				voxel: float,
				crop_pts: np.ndarray,
				crop_cols: np.ndarray | None,
				crop_vox_pts: np.ndarray,
				crop_vox_cols: np.ndarray | None,
				fmt: str = "pcd") -> tuple[Path, Path]:
	out_dir.mkdir(parents=True, exist_ok=True)
	side_s = _sanitize_float(side)
	h_s = _sanitize_float(height_mult)
	v_s = _sanitize_float(voxel)
	crop_name = f"{in_stem}_crop_{plane}_s{side_s}_h{h_s}.{fmt}"
	crop_vox_name = f"{in_stem}_crop_vox_{plane}_s{side_s}_h{h_s}_v{v_s}.{fmt}"
	crop_path = out_dir / crop_name
	crop_vox_path = out_dir / crop_vox_name

	crop_pc = _to_o3d_cloud(crop_pts, crop_cols)
	crop_vox_pc = _to_o3d_cloud(crop_vox_pts, crop_vox_cols)

	ok1 = o3d.io.write_point_cloud(str(crop_path), crop_pc)
	ok2 = o3d.io.write_point_cloud(str(crop_vox_path), crop_vox_pc)
	if not ok1:
		raise RuntimeError(f"Failed to write {crop_path}")
	if not ok2:
		raise RuntimeError(f"Failed to write {crop_vox_path}")
	return crop_path, crop_vox_path

DEFAULT_PCD="SNOX-Selecao/pc/entire.pcd"
def main() -> None:
	parser = argparse.ArgumentParser(description="Cut a square from a PCD around its centroid, voxelize, and view in rerun.io")
	parser.add_argument("--pcd", type=str, default=DEFAULT_PCD, help="Path to the .pcd file. If omitted, auto-picks one under dataset/pcds/")
	parser.add_argument("--plane", type=str, choices=["xy", "yz", "xz"], default="xy", help="Plane on which the square is defined")
	parser.add_argument("--side", type=float, default=None, help="Side length of the square. Default is ratio * min(range_along_plane)")
	parser.add_argument("--ratio", type=float, default=0.06, help="When --side is not given, use this fraction of the smaller plane span (0-1]")
	parser.add_argument("--height-mult", type=float, default=2.0, help="Height multiplier along axis orthogonal to plane (height = multiplier * side)")
	parser.add_argument("--voxel", type=float, default=0.01, help="Voxel size for downsampling the cropped cloud")
	parser.add_argument("--out-dir", type=str, default="outputs", help="Directory to save cropped clouds")
	parser.add_argument("--out-fmt", type=str, choices=["pcd", "ply"], default="pcd", help="File format for saved clouds")
	parser.add_argument("--no-save", action="store_true", help="Do not save cropped clouds to disk")
	parser.add_argument("--dry-run", action="store_true", help="Do not open rerun viewer; just print counts")
	args = parser.parse_args()

	pcd_path = args.pcd or _auto_find_pcd()
	if not pcd_path:
		raise SystemExit("No PCD file provided and none found under dataset/pcds/**. Please pass --pcd <path>.")

	pc = load_point_cloud(pcd_path)
	pts, cols = to_numpy(pc)
	ctr = compute_centroid(pts)

	side = args.side if args.side is not None else infer_side_length(pts, args.plane, args.ratio)
	if side <= 0:
		raise SystemExit("Computed/Provided side length must be positive.")

	mask = crop_square_mask(pts, ctr, args.plane, side, args.height_mult)
	crop_pts = pts[mask]
	crop_cols = cols[mask] if cols is not None else None

	# Voxel downsample the cropped cloud
	crop_vox_pts, crop_vox_cols = voxel_downsample(crop_pts, crop_cols, args.voxel)

	# Save both cropped and voxelized clouds unless disabled
	out_dir = Path(args.out_dir)
	in_stem = Path(pcd_path).stem
	crop_path = None
	crop_vox_path = None
	if not args.no_save:
		crop_path, crop_vox_path = save_clouds(out_dir, in_stem, args.plane, side, args.height_mult, args.voxel,
														 crop_pts, crop_cols, crop_vox_pts, crop_vox_cols, fmt=args.out_fmt)

	if args.dry_run:
		total = pts.shape[0]
		kept = crop_pts.shape[0]
		kept_vox = crop_vox_pts.shape[0]
		frac = kept / total if total else 0.0
		frac_vox = kept_vox / total if total else 0.0
		print(f"PCD: {pcd_path}")
		print(f"Plane: {args.plane} | Side: {side:.4f} | HeightMult: {args.height_mult:.2f} | Voxel: {args.voxel:.4f}")
		print(f"Cropped: {kept}/{total} ({frac:.1%}) | Voxelized crop: {kept_vox} ({frac_vox:.1%} of total)")
		if not args.no_save:
			print(f"Saved: {crop_path}")
			print(f"Saved: {crop_vox_path}")
		return

	# Only visualize the voxelized cropped cloud
	visualize_rerun(crop_vox_pts, crop_vox_cols)


if __name__ == "__main__":
	main()

