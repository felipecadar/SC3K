#!/usr/bin/env python3
"""
Script to generate SC3K dataset from a folder containing .pcd files.
Generates random poses for each point cloud and creates the required dataset structure.
"""

import os
import shutil
import numpy as np
import argparse
import json
from glob import glob
from sklearn.model_selection import train_test_split
import open3d as o3d
from pathlib import Path
from tqdm import tqdm


def generate_random_pose():
    """Generate a random 3x4 pose matrix (rotation + translation)."""
    # Random rotation matrix
    rotation = _random_rotation_matrix()
    
    # Random translation (reasonable range for point clouds)
    translation = np.random.uniform(-2, 2, (3, 1))
    
    # Combine into 3x4 matrix
    pose = np.hstack([rotation, translation])
    return pose


def _random_rotation_matrix():
    """Generate a random 3x3 rotation matrix."""
    u1, u2, u3 = np.random.rand(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    
    x, y, z, w = q1, q2, q3, q4
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def divide_bbox(large_bbox: o3d.geometry.AxisAlignedBoundingBox, side_length: float, overlap: float = 0.0):
    """Divide a big AABB into overlapping cubic tiles."""
    if side_length <= 0:
        print(f"Error: side_length for divide_bbox must be positive.")
        return []
    if not 0 <= overlap < 1:
        print(f"Error: overlap must be in the range [0, 1).")
        return []

    min_bound = large_bbox.min_bound
    max_bound = large_bbox.max_bound
    extent = large_bbox.get_extent()
    stride = side_length * (1 - overlap)
    if stride <= 1e-9:
        if any(extent > side_length):
            print(f"Error: Overlap is too high, resulting in zero stride.")
        return [large_bbox]

    num_div_x = int(np.ceil(max(0, extent[0] - side_length) / stride)) + 1
    num_div_y = int(np.ceil(max(0, extent[1] - side_length) / stride)) + 1
    num_div_z = int(np.ceil(max(0, extent[2] - side_length) / stride)) + 1

    out = []
    for i in range(num_div_x):
        for j in range(num_div_y):
            for k in range(num_div_z):
                cur_min = np.array([
                    min_bound[0] + i * stride,
                    min_bound[1] + j * stride,
                    min_bound[2] + k * stride,
                ], dtype=np.float32)
                cur_max = cur_min + side_length
                cur_max = np.minimum(cur_max, max_bound)
                small = o3d.geometry.AxisAlignedBoundingBox(cur_min, cur_max)
                if small.volume() > 1e-9:
                    out.append(small)
    return out


def split_pcd_into_cubes(pcd_path: str, cube_size: float = 1.0, overlap: float = 0.0, output_dir: str = "parts", min_points: int = 2048):
    """Split a large PCD file into smaller cubes of fixed size."""
    print(f"Loading PCD file: {pcd_path}")
    
    # Load point cloud
    pc = o3d.io.read_point_cloud(pcd_path)
    if pc.is_empty():
        print(f"Warning: Point cloud {pcd_path} is empty")
        return []
    
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors) if pc.has_colors() else None
    
    print(f"Original point cloud has {len(points)} points")
    
    # Create AABB for the entire point cloud
    aabb = pc.get_axis_aligned_bounding_box()
    print(f"Point cloud bounds: {aabb.min_bound} to {aabb.max_bound}")
    
    # Divide into smaller cubes
    small_aabbs = divide_bbox(aabb, cube_size, overlap)
    print(f"Dividing into {len(small_aabbs)} cubes of size {cube_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    part_files = []
    skipped_cubes = 0
    base_name = Path(pcd_path).stem
    
    for i, small_aabb in enumerate(tqdm(small_aabbs, desc=f"Processing cubes for {base_name}")):
        # Crop point cloud using Open3D's crop_point_cloud method
        cropped_pc = pc.crop(small_aabb)
        
        # Check if cropped point cloud has points
        cropped_points = np.asarray(cropped_pc.points)
        num_points = len(cropped_points)
        
        print(f"Cube {i}: bounds {small_aabb.min_bound} to {small_aabb.max_bound}, points: {num_points}")
        
        if num_points == 0:
            skipped_cubes += 1
            continue
        
        # Skip cubes with too few points
        if num_points < min_points:
            print(f"Skipping cube {i}: only {num_points} points (minimum: {min_points})")
            skipped_cubes += 1
            continue
        
        # Save the cropped part
        part_filename = f"{base_name}_part_{i:04d}.pcd"
        part_path = os.path.join(output_dir, part_filename)
        o3d.io.write_point_cloud(part_path, cropped_pc)
        
        part_files.append(part_path)
        print(f"Saved part {i}: {num_points} points -> {part_filename}")
    
    print(f"Created {len(part_files)} parts from {pcd_path} ({skipped_cubes} cubes skipped)")
    return part_files


def create_simple_annotation_template(model_ids, category_id="02691156", include_semantic_id=False):
    """Create a simple annotation template with basic keypoints."""
    annotations = []
    
    for model_id in model_ids:
        # Create a simple annotation with a few keypoints
        keypoints = []
        for i in range(10):  # 10 keypoints
            keypoint = {
                "xyz": [np.random.uniform(-1, 1) for _ in range(3)],
                "rgb": [255, 255, 255],
                "pcd_info": {"point_index": np.random.randint(0, 1000)},
                "mesh_info": {
                    "face_index": np.random.randint(0, 10000),
                    "face_uv": [np.random.random(), np.random.random(), np.random.random()]
                }
            }
            
            # Only include semantic_id if requested
            if include_semantic_id:
                keypoint["semantic_id"] = i
                
            keypoints.append(keypoint)
        
        annotation = {
            "class_id": category_id,
            "model_id": model_id,
            "keypoints": keypoints,
            "symmetries": {
                "reflection": [],
                "rotation": []
            }
        }
        annotations.append(annotation)
    
    return annotations


def crop_center_square(pc: o3d.geometry.PointCloud, crop_size: float, center: np.ndarray = None):
    """Crop a square region from the center of the point cloud."""
    points = np.asarray(pc.points)
    
    if center is None:
        # Use the centroid of the point cloud
        center = np.mean(points, axis=0)
    
    # Define the cropping bounds (square)
    min_bound = center - crop_size / 2
    max_bound = center + crop_size / 2
    
    # Create axis-aligned bounding box for cropping
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    # Crop the point cloud
    cropped_pc = pc.crop(aabb)
    
    print(f"Cropped to center square: {len(np.asarray(cropped_pc.points))} points "
          f"(from {len(points)}) around center {center}")
    
    return cropped_pc


def apply_initial_processing(pc: o3d.geometry.PointCloud, voxel_size: float = None, 
                           crop_size: float = None, crop_center: np.ndarray = None):
    """Apply initial processing: voxel downsampling and/or center cropping."""
    original_points = len(np.asarray(pc.points))
    
    # Apply voxel downsampling if requested
    if voxel_size is not None and voxel_size > 0:
        pc = pc.voxel_down_sample(voxel_size=voxel_size)
        print(f"Initial voxel downsampling: {len(np.asarray(pc.points))} points "
              f"(from {original_points}) with voxel size {voxel_size}")
    
    # Apply center cropping if requested
    if crop_size is not None and crop_size > 0:
        pc = crop_center_square(pc, crop_size, crop_center)
    
    return pc


def main(input_folder, output_folder, category_id="02691156", category_name="airplane", 
         cube_size=1.0, overlap=0.0, split_large_pcds=True, initial_voxel_size=None,
         crop_size=None, crop_center=None, min_points=2048):
    """Generate dataset from .pcd files."""
    
    # Initialize rerun for visualization
    try:
        import rerun as rr
        rr.init("Dataset Generation Viewer")
        rr.spawn()
        use_rerun = True
    except ImportError:
        print("Warning: rerun not available, skipping visualizations")
        use_rerun = False
    
    # Find all .pcd files
    pcd_files = glob(os.path.join(input_folder, "*.pcd"))
    if not pcd_files:
        print(f"No .pcd files found in {input_folder}")
        return
    
    print(f"Found {len(pcd_files)} .pcd files")
    
    # Split large PCDs into parts if requested
    all_pcd_files = []
    parts_dir = os.path.join(output_folder, "parts")
    
    if split_large_pcds:
        print("Splitting large PCD files into cubes...")
        for pcd_file in tqdm(pcd_files, desc="Processing PCD files"):
            print(f"Processing {pcd_file}")
            
            # Load original point cloud
            pc = o3d.io.read_point_cloud(pcd_file)
            if pc.is_empty():
                print(f"Warning: Point cloud {pcd_file} is empty, skipping")
                continue
            
            # Apply initial processing (voxel downsampling and/or cropping)
            pc = apply_initial_processing(pc, initial_voxel_size, crop_size, crop_center)
            
            # Save the processed version temporarily for splitting
            temp_processed_path = os.path.join(parts_dir, f"temp_{os.path.basename(pcd_file)}")
            os.makedirs(parts_dir, exist_ok=True)
            o3d.io.write_point_cloud(temp_processed_path, pc)
            
            # Split the processed point cloud into cubes
            part_files = split_pcd_into_cubes(temp_processed_path, cube_size, overlap, parts_dir, min_points)
            all_pcd_files.extend(part_files)
            
            # Clean up temporary file
            if os.path.exists(temp_processed_path):
                os.remove(temp_processed_path)
                
        print(f"Total parts created: {len(all_pcd_files)}")
    else:
        # If not splitting, still apply initial processing
        if initial_voxel_size is not None or crop_size is not None:
            print("Applying initial processing to PCD files...")
            processed_dir = os.path.join(output_folder, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            for pcd_file in tqdm(pcd_files, desc="Processing PCD files"):
                pc = o3d.io.read_point_cloud(pcd_file)
                if pc.is_empty():
                    print(f"Warning: Point cloud {pcd_file} is empty, skipping")
                    continue
                
                pc = apply_initial_processing(pc, initial_voxel_size, crop_size, crop_center)
                
                processed_path = os.path.join(processed_dir, os.path.basename(pcd_file))
                o3d.io.write_point_cloud(processed_path, pc)
                all_pcd_files.append(processed_path)
        else:
            all_pcd_files = pcd_files
    
    if not all_pcd_files:
        print("No valid PCD files to process")
        return
    
    # Create dataset structure
    os.makedirs(os.path.join(output_folder, "pcds", category_id), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "poses", category_id), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "splits"), exist_ok=True)
    
    # Extract model IDs from filenames and process each model
    model_ids = []
    for pcd_file in tqdm(all_pcd_files, desc="Generating poses"):
        model_id = os.path.splitext(os.path.basename(pcd_file))[0]
        model_ids.append(model_id)
        
        # Copy .pcd file
        dest_pcd = os.path.join(output_folder, "pcds", category_id, f"{model_id}.pcd")
        shutil.copy2(pcd_file, dest_pcd)
        
        # Load and visualize the point cloud if rerun is available
        if use_rerun:
            try:
                pc = o3d.io.read_point_cloud(pcd_file)
                if not pc.is_empty():
                    # Create voxel grid version for visualization (better performance)
                    voxel_size = 0.01  # 1cm voxels for good balance of detail and performance
                    pc_downsampled = pc.voxel_down_sample(voxel_size=voxel_size)
                    
                    pts = np.asarray(pc_downsampled.points)
                    cols = None
                    if pc_downsampled.has_colors():
                        cols = (np.asarray(pc_downsampled.colors) * 255.0).astype(np.uint8)
                        if cols.ndim == 2 and cols.shape[1] == 3:
                            alpha = np.full((cols.shape[0], 1), 255, dtype=np.uint8)
                            cols = np.concatenate([cols, alpha], axis=1)
                    else:
                        # Default color
                        cols = np.full((pts.shape[0], 4), [255, 255, 255, 255], dtype=np.uint8)
                    
                    print(f"Visualizing {model_id}: {len(pts)} points (downsampled from {len(np.asarray(pc.points))} for performance)")
                    rr.log(f"dataset/{model_id}", rr.Points3D(positions=pts, colors=cols, radii=0.003))
                    
                    # Log bounding box
                    bbx = pc.get_axis_aligned_bounding_box()  # Use original PC for accurate bbox
                    center = bbx.get_center()
                    half_size = bbx.get_extent() / 2
                    rr.log(f"dataset/{model_id}_bbox", rr.Boxes3D(
                        centers=np.array([center]),
                        half_sizes=np.array([half_size])
                    ))
            except Exception as e:
                print(f"Warning: Could not visualize {pcd_file}: {e}")
        
        # Generate 24 random poses
        pose_data = {}
        for i in range(24):
            pose = generate_random_pose()
            pose_data[f'world_mat_{i}'] = pose.astype(np.float32)
            pose_data[f'camera_mat_{i}'] = pose.astype(np.float32)  # Using same for simplicity
        
        # Save poses
        pose_file = os.path.join(output_folder, "poses", category_id, f"{model_id}.npz")
        np.savez(pose_file, **pose_data)
    
    print(f"Generated poses for {len(model_ids)} models")
    
    # Create simple annotations
    annotations = create_simple_annotation_template(model_ids, category_id, args.include_semantic_id)
    annot_file = os.path.join(output_folder, "annotations", f"{category_name}.json")
    with open(annot_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    # Create splits (80% train, 10% val, 10% test)
    if len(model_ids) < 3:
        # If we have very few samples, put them all in train
        train_ids = model_ids
        val_ids = []
        test_ids = []
        print(f"Warning: Only {len(model_ids)} models found. Putting all in training set.")
    else:
        train_ids, temp_ids = train_test_split(model_ids, test_size=0.2, random_state=42)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    
    # Write split files
    def write_split_file(ids, split_name):
        with open(os.path.join(output_folder, "splits", f"{split_name}.txt"), 'w') as f:
            for model_id in ids:
                f.write(f"{category_id}-{model_id}\n")
    
    write_split_file(train_ids, "train")
    write_split_file(val_ids, "val")
    write_split_file(test_ids, "test")
    
    print(f"Created splits: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    print(f"Dataset generated in {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SC3K dataset from .pcd files")
    parser.add_argument("input_folder", help="Folder containing .pcd files")
    parser.add_argument("output_folder", help="Output folder for dataset")
    parser.add_argument("--category_id", default="02691156", help="Category ID (default: plataform)")
    parser.add_argument("--category_name", default="plataform", help="Category name")
    parser.add_argument("--cube_size", type=float, default=1.0, help="Size of cubes for splitting large PCDs")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap between cubes (0-1)")
    parser.add_argument("--no_split", action="store_true", help="Don't split large PCDs into parts")
    parser.add_argument("--initial_voxel_size", type=float, default=None, 
                       help="Apply voxel downsampling at the start with given voxel size")
    parser.add_argument("--crop_size", type=float, default=None,
                       help="Crop a square region of given size from the center")
    parser.add_argument("--crop_center", type=float, nargs=3, default=None,
                       help="Center point for cropping (x y z), defaults to centroid")
    parser.add_argument("--min_points", type=int, default=2048,
                       help="Minimum number of points required per cube (default: 2048)")
    parser.add_argument("--include_semantic_id", action="store_true",
                       help="Include semantic_id in annotations (default: False)")
    
    args = parser.parse_args()
    
    # Parse crop_center if provided
    crop_center = None
    if args.crop_center:
        crop_center = np.array(args.crop_center)
    
    main(args.input_folder, args.output_folder, args.category_id, args.category_name,
         args.cube_size, args.overlap, not args.no_split, args.initial_voxel_size,
         args.crop_size, crop_center, args.min_points)
