"""
Run SC3K keypoint detection on a single PCD file.

Example:
  # Default to 10 keypoints, normalize input, and save a PNG/PLY visualization
  uv run python run_on_pcd.py --pcd P68/P68_scan.pcd --weights /path/to/Best_airplane_10kp.pth --save-vis

Notes:
- Torch must be installed in your environment (see environment.yml for recommended versions).
- The number of keypoints must match the weights you provide (default: 10).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List, Tuple

import numpy as np
import torch
import open3d as o3d

import network  # uses torch
from data_loader import normalize_pc, farthest_point_sample
import visualizations as viz


def load_pcd_points(pcd_path: str, downsample=0.01) -> np.ndarray:
    p = Path(pcd_path)
    if not p.exists():
        raise FileNotFoundError(f"PCD not found: {pcd_path}")
    pc = o3d.io.read_point_cloud(str(p))
    # voxel downsample if requested
    if downsample and downsample > 0:
        pc = pc.voxel_down_sample(voxel_size=downsample)
    if pc.is_empty():
        raise ValueError(f"Point cloud is empty: {pcd_path}")
    pts = np.asarray(pc.points, dtype=np.float32)
    return pts


def prepare_points(
    pts: np.ndarray,
    normalize: bool = True,
    sample_points: Optional[int] = None,
) -> np.ndarray:
    if normalize:
        pts = normalize_pc(pts)
    if sample_points and sample_points > 0 and pts.shape[0] > sample_points:
        pts = farthest_point_sample(pts, sample_points)
    return pts.astype(np.float32)


def normalize_with_params(pts: np.ndarray, enabled: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize like data_loader.normalize_pc but also return mean and scale for inverse mapping."""
    if not enabled:
        return pts.astype(np.float32), np.zeros(3, dtype=np.float32), 1.0
    mean = pts.mean(axis=0)
    centered = pts - mean
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale <= 0:
        scale = 1.0
    pts_norm = (centered / scale).astype(np.float32)
    return pts_norm, mean.astype(np.float32), scale


def divide_bbox(large_bbox: o3d.geometry.AxisAlignedBoundingBox, side_length: float, overlap: float = 0.0) -> List[o3d.geometry.AxisAlignedBoundingBox]:
    """Divide a big AABB into overlapping cubic tiles.

    Implementation adapted from user's reference.
    """
    if side_length <= 0:
        print("[bold red]Error: side_length for divide_bbox must be positive.[/bold red]")
        return []
    if not 0 <= overlap < 1:
        print("[bold red]Error: overlap must be in the range [0, 1).[/bold red]")
        return []

    min_bound = large_bbox.min_bound
    max_bound = large_bbox.max_bound
    extent = large_bbox.get_extent()
    stride = side_length * (1 - overlap)
    if stride <= 1e-9:
        if any(extent > side_length):
            print("[bold red]Error: Overlap is too high, resulting in zero stride.[/bold red]")
        return [large_bbox]

    num_divisions_x = int(np.ceil(max(0, extent[0] - side_length) / stride)) + 1
    num_divisions_y = int(np.ceil(max(0, extent[1] - side_length) / stride)) + 1
    num_divisions_z = int(np.ceil(max(0, extent[2] - side_length) / stride)) + 1

    smaller_bboxes: List[o3d.geometry.AxisAlignedBoundingBox] = []
    for i in range(num_divisions_x):
        for j in range(num_divisions_y):
            for k in range(num_divisions_z):
                current_min_x = min_bound[0] + i * stride
                current_min_y = min_bound[1] + j * stride
                current_min_z = min_bound[2] + k * stride
                small_min = np.array([current_min_x, current_min_y, current_min_z], dtype=np.float32)

                current_max_x = current_min_x + side_length
                current_max_y = current_min_y + side_length
                current_max_z = current_min_z + side_length

                final_max_x = min(current_max_x, max_bound[0])
                final_max_y = min(current_max_y, max_bound[1])
                final_max_z = min(current_max_z, max_bound[2])
                small_max = np.array([final_max_x, final_max_y, final_max_z], dtype=np.float32)

                small_bbox = o3d.geometry.AxisAlignedBoundingBox(small_min, small_max)
                if small_bbox.volume() > 1e-9:
                    smaller_bboxes.append(small_bbox)
    return smaller_bboxes


def inside_aabb_mask(points: np.ndarray, aabb: o3d.geometry.AxisAlignedBoundingBox) -> np.ndarray:
    minb = aabb.min_bound
    maxb = aabb.max_bound
    return (
        (points[:, 0] >= minb[0]) & (points[:, 0] <= maxb[0]) &
        (points[:, 1] >= minb[1]) & (points[:, 1] <= maxb[1]) &
        (points[:, 2] >= minb[2]) & (points[:, 2] <= maxb[2])
    )


def save_vis_open3d(pc: np.ndarray, kps: np.ndarray, outdir: Path, name: str = "combined") -> None:
    """Save a simple PLY+PNG visual using Open3D, robust to many keypoints."""
    palette_pc = (0.2, 0.2, 0.2)
    kp_color = (1.0, 0.2, 0.2)

    mesh = o3d.geometry.TriangleMesh()
    # Points
    for p in pc:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        sph.translate(p.astype(float))
        sph.paint_uniform_color(palette_pc)
        mesh += sph
    # Keypoints
    for q in kps:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)
        sph.translate(q.astype(float))
        sph.paint_uniform_color(kp_color)
        mesh += sph

    (outdir / "ply").mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(outdir / "ply" / f"{name}.ply"), mesh)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    vis.poll_events(); vis.update_renderer()
    (outdir / "png").mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(outdir / "png" / f"{name}.png"))
    vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SC3K on a single PCD file")
    parser.add_argument("--pcd", required=True, help="Path to .pcd file")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--keypoints", type=int, default=64, help="Number of keypoints (must match weights)")
    parser.add_argument("--no-normalize", action="store_true", help="Disable point cloud normalization")
    parser.add_argument("--sample", type=int, default=0, help="Optional farthest point sampling count (0 = no sampling)")
    parser.add_argument("--device", default="auto", help="Device to run on")
    parser.add_argument("--save-vis", action="store_true", help="Save visualization (PLY+PNG)")
    parser.add_argument("--outdir", default="outputs/run_on_pcd", help="Output directory for artifacts")
    parser.add_argument("--export", default=None, help="Optional path to save keypoints (.npy or .txt)")
    parser.add_argument("--rerun", action="store_true", help="Also log to rerun viewer")
    # Chunking options
    parser.add_argument("--chunk-size", type=float, default=0.0, help="If > 0, split into cubic chunks of this side length (in input units)")
    parser.add_argument("--chunk-overlap", type=float, default=0.0, help="Overlap fraction between chunks [0,1)")
    parser.add_argument("--min-chunk-points", type=int, default=100, help="Skip chunks with fewer points than this")
    parser.add_argument("--chunk-limit", type=int, default=0, help="Optionally process only the first N chunks (0 = all)")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build minimal config for the model
    cfg = SimpleNamespace()
    cfg.key_points = int(args.keypoints)
    cfg.task = "canonical"  # single-cloud inference path
    cfg.split = "test"

    # Load network and weights
    model = network.sc3k(cfg).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Load the full point cloud
    pts_full = load_pcd_points(args.pcd)

    combined_kps: List[np.ndarray] = []
    stem = Path(args.pcd).stem
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    

    if args.chunk_size and args.chunk_size > 0:
        # Compute AABB and tiles
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(pts_full.astype(float))
        big_aabb = pc_o3d.get_axis_aligned_bounding_box()
        tiles = divide_bbox(big_aabb, side_length=float(args.chunk_size), overlap=float(args.chunk_overlap))
        print(f"Chunking enabled: {len(tiles)} tiles computed")

        processed = 0
        for idx, tile in enumerate(tiles):
            mask = inside_aabb_mask(pts_full, tile)
            pts_chunk = pts_full[mask]
            if pts_chunk.shape[0] < args.min_chunk_points:
                continue

            # Optional sampling
            if args.sample and args.sample > 0 and pts_chunk.shape[0] > args.sample:
                pts_chunk = farthest_point_sample(pts_chunk.astype(np.float32), args.sample)

            # Normalize per-chunk and remember params
            pts_norm, mean, scale = normalize_with_params(pts_chunk, enabled=not args.no_normalize)

            # Forward pass
            pc_tensor = torch.from_numpy(pts_norm[None, ...]).to(device)
            with torch.no_grad():
                pred = model([pc_tensor])  # [1,K,3]
            kp_norm = pred[0].detach().cpu().numpy()

            # Invert normalization to global coords
            kp_global = kp_norm * float(scale) + mean
            combined_kps.append(kp_global)
            processed += 1

            if args.chunk_limit and processed >= args.chunk_limit:
                break

        if not combined_kps:
            print("No chunks produced keypoints (maybe min-chunk-points too high?).")
            return
        kp_all = np.concatenate(combined_kps, axis=0)
        print(f"Detected {kp_all.shape[0]} keypoints over {processed} chunks for '{stem}'.")
    else:
        # Single pass on the full cloud
        pts_pre = prepare_points(pts_full, normalize=not args.no_normalize, sample_points=args.sample)
        pc_tensor = torch.from_numpy(pts_pre[None, ...]).to(device)
        with torch.no_grad():
            pred = model([pc_tensor])  # [1,K,3]
        kp_all = pred[0].detach().cpu().numpy()
        print(f"Detected {kp_all.shape[0]} keypoints for '{stem}'.")

    # Ensure output dir
    # Export keypoints
    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if export_path.suffix.lower() == ".npy":
            np.save(str(export_path), kp_all)
        else:
            # Save as whitespace-separated text by default
            np.savetxt(str(export_path), kp_all, fmt="%.6f")
        print(f"Saved keypoints to: {export_path}")
    else:
        # Default export: outdir/<name>_kp.npy
        np.save(str(outdir / f"{stem}_kp.npy"), kp_all)

    # Optional visualization
    if args.save_vis:
        if args.chunk_size and args.chunk_size > 0:
            # Use robust custom visualizer for many keypoints
            save_vis_open3d(pts_full, kp_all, outdir, name=stem)
        else:
            viz.save_kp_and_pc_in_pcd(pts_pre, kp_all, str(outdir), save=True, name=stem)
        print(f"Saved visualization (PLY+PNG) under: {outdir}")

    # Optional rerun logging
    if args.rerun:
        import seaborn as sns
        palette = sns.color_palette("bright", n_colors=max(10, len(kp_all)))
        try:
            import rerun as rr
        except Exception as e:
            print(
                "Failed to import rerun. To enable --rerun, ensure rerun-sdk is installed and you're on Python >= 3.9.\n"
                f"Import error: {e}"
            )
        else:
            rr.init("SC3K Inference")
            rr.spawn()

            cloud_color = np.array([200, 200, 200, 255], dtype=np.uint8)
            if args.chunk_size and args.chunk_size > 0:
                # Per-chunk colors
                # palette = [
                #     (255, 99, 99, 255), (99, 181, 255, 255), (99, 255, 148, 255),
                #     (255, 214, 99, 255), (207, 99, 255, 255), (99, 255, 240, 255),
                #     (255, 140, 140, 255), (140, 200, 255, 255), (140, 255, 200, 255),
                # ]
                rr.log(f"cloud/{stem}", rr.Points3D(positions=pts_full, colors=np.repeat(cloud_color[None, :], pts_full.shape[0], axis=0), radii=0.003))
                # Log combined KPs with a single layer; optionally could log per-chunk
                kp_colors = np.repeat(np.array(palette[0], dtype=np.uint8)[None, :], kp_all.shape[0], axis=0)
                rr.log(f"keypoints/{stem}", rr.Points3D(positions=kp_all, colors=kp_colors, radii=0.01))
            else:
                cloud_colors = np.repeat(cloud_color[None, :], pts_pre.shape[0], axis=0)
                kp_color = np.array([255, 64, 64, 255], dtype=np.uint8)
                kp_colors = np.repeat(kp_color[None, :], kp_all.shape[0], axis=0)
                rr.log(f"cloud/{stem}", rr.Points3D(positions=pts_pre, colors=cloud_colors, radii=0.003))
                rr.log(f"keypoints/{stem}", rr.Points3D(positions=kp_all, colors=kp_colors, radii=0.01))
            print("Logged to rerun viewer. If no window opened, ensure a GUI is available or use the rerun app.")


if __name__ == "__main__":
    main()
