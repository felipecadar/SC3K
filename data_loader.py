import numpy as np
from glob import glob
import os
import json
from torchvision import transforms
import pdb
import hydra
import torch
import omegaconf
from tqdm import tqdm

import open3d as o3d
import itertools    # join lists of list in one_list
import matplotlib.pyplot as plt

BASEDIR = os.path.dirname(os.path.abspath(__file__))
ID2NAMES = {"02691156": "airplane",
            "02808440": "bathtub",
            "02818832": "bed",
            "02876657": "bottle",
            "02954340": "cap",
            "02958343": "car",
            "03001627": "chair",
            "03467517": "guitar",
            "03513137": "helmet",
            "03624134": "knife",
            "03642806": "laptop",
            "03790512": "motorcycle",
            "03797390": "mug",
            "04225987": "skateboard",
            "04379243": "table",
            "04530566": "vessel", 
            "44444444": "plataform",
        }

NAMES2ID = {v: k for k, v in ID2NAMES.items()}


def naive_read_pcd(path):
    # Use Open3D to read PCD files (handles both ASCII and binary formats)
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if colors.size == 0:
        # If no colors, create white colors
        colors = np.ones((points.shape[0], 3), dtype=np.uint8) * 255
    else:
        # Convert from float [0,1] to uint8 [0,255]
        colors = (colors * 255).astype(np.uint8)
    return points, colors


def add_noise(x, sigma=0.015, clip=0.05):
    noise = np.clip(sigma * np.random.randn(*x.shape), -1 * clip, clip)
    return x + noise


def normalize_pc(pc):
    pc = pc - pc.mean(0)
    pc /= np.max(np.linalg.norm(pc, axis=-1))
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def sample_fixed_points(points: np.ndarray, target: int) -> np.ndarray:
    """Return exactly `target` points.
    - If more: farthest point sample.
    - If fewer: sample with replacement to upsample.
    """
    n = points.shape[0]
    if n == target:
        return points.astype(np.float32)
    if n > target:
        pts = points.astype(np.float32)
        # Pre-reduce very large sets to cap FPS cost
        cap_factor = 4
        cap = target * cap_factor
        if n > cap:
            idx = np.random.choice(n, size=cap, replace=False)
            pts = pts[idx]
        return farthest_point_sample(pts, target)
    # n < target: upsample with replacement
    idx = np.random.choice(n, size=target, replace=True)
    return points[idx].astype(np.float32)


def transform(pc, extrinsic_mat):
    zup = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype='f')  # Z_UP
    return np.dot(extrinsic_mat @ zup, pc.T).T


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _random_rotation_matrix() -> np.ndarray:
    """Generate a random 3x3 rotation matrix (right-handed, det=1).

    Uses uniform random unit quaternion sampling.
    """
    u1, u2, u3 = np.random.rand(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    # quaternion to rotation matrix (q = [q1,q2,q3,q4] with q4 as w)
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


def inside_aabb_mask(points: np.ndarray, aabb: o3d.geometry.AxisAlignedBoundingBox) -> np.ndarray:
    minb = aabb.min_bound
    maxb = aabb.max_bound
    return (
        (points[:, 0] >= minb[0]) & (points[:, 0] <= maxb[0]) &
        (points[:, 1] >= minb[1]) & (points[:, 1] <= maxb[1]) &
        (points[:, 2] >= minb[2]) & (points[:, 2] <= maxb[2])
    )


class generic_data_loader(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.catg = cfg.class_name
        self.cfg = cfg
        self.cat = []
        self.cat.append(NAMES2ID[cfg.class_name])

        annots = json.load(open(os.path.join(BASEDIR, cfg.data.annot_path)))
        annots = [annot for annot in annots if annot['class_id'] in self.cat]

        selected_cat = []
        for i in range(len(annots)):
            if annots[i]['class_id'] not in selected_cat:
                selected_cat.append(annots[i]['class_id'])
        print('loaded {} samples of categories: '.format(len(annots)), selected_cat)

        pcd_paths_np = []
        for i in range(len(selected_cat)):
            pcd_paths_np += glob(os.path.join(BASEDIR, cfg.data.pcd_root, selected_cat[i], '*.pcd'))

        # Calculate nclasses - handle case where semantic_id might not exist (self-supervised learning)
        try:
            self.nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1
        except KeyError:
            # For self-supervised learning, set nclasses to 1 since semantic classes are not used
            self.nclasses = 1

        split_models = open(os.path.join(BASEDIR, cfg.data.splits_root, "{}.txt".format(split))).readlines()
        split_models = [m.split('-', 1)[1].rstrip('\n') for m in split_models]

        mesh_names = []
        camera_param_np = []
        camera_param_np_2 = []
        pointCloud_lst = []
        pointCloud_lst_2 = []
        print("Loading {} data, please wait\n".format(split))
        for fn in tqdm(pcd_paths_np):
            model_id = os.path.basename(fn).split('.')[0]
            if model_id not in split_models:
                continue

            cat_name = fn.split('/')[-2]
            mesh_names.append(model_id)

            pc_list = []
            cam_lst = []
            camera_mat = np.load(os.path.join(BASEDIR, cfg.data.poses_root, cat_name, '{}.npz'.format(model_id)))
            for i in range(24):
                cam_lst.append(camera_mat['world_mat_{}'.format(i)][:,:3])
                pc_list.append(transform(naive_read_pcd(fn)[0], camera_mat['world_mat_{}'.format(i)][:,:3]))
            camera_param_np.append(cam_lst)
            camera_param_np_2.append(cam_lst[::-1])
            pointCloud_lst.append(pc_list)
            pointCloud_lst_2.append(pc_list[::-1])

        self.transformed_pcds = pointCloud_lst
        self.transformed_pcds_2 = pointCloud_lst_2
        self.camera_param = camera_param_np
        self.camera_param_2 = camera_param_np_2
        self.mesh_names = mesh_names

        print("\n\nPlease wait, arranging the data\n\n")
        self.camera_param_np = list(itertools.chain.from_iterable(camera_param_np))     # combine array elements in
        self.camera_param_np_2 = list(itertools.chain.from_iterable(camera_param_np_2))  # combine array elements in
        self.transformed_pcds = list(itertools.chain.from_iterable(pointCloud_lst))  # combine array elements in
        self.transformed_pcds_2 = list(itertools.chain.from_iterable(pointCloud_lst_2))  # combine array elements in
        self.mesh_names = list(np.repeat(mesh_names, 24))     # repeat list

        print("\n\nloaded data contains: ")
        print("  * camera_param 1: {}".format(len(self.camera_param_np)))
        print("  * camera_param 2: {}".format(len(self.camera_param_np_2)))
        print("  * transformed_pcds 1: {}".format(len(self.transformed_pcds)))
        print("  * transformed_pcds 2: {}".format(len(self.transformed_pcds_2)))
        print("  * mesh_names: {}\n\n".format(len(self.mesh_names)))

    def __getitem__(self, idx):
        pcd1 = self.transformed_pcds[idx]
        pcd2 = self.transformed_pcds_2[idx]
        camera_matrix = self.camera_param_np[idx]
        camera_matrix2 = self.camera_param_np_2[idx]
        mesh_name = self.mesh_names[idx]

        if self.cfg.augmentation.normalize_pc:
            pcd1 = normalize_pc(pcd1)
            pcd2 = normalize_pc(pcd2)

        if self.cfg.augmentation.down_sample:
            pcd1 = farthest_point_sample(pcd1, self.cfg.sample_points)
            pcd2 = farthest_point_sample(pcd2, self.cfg.sample_points)

        if self.cfg.augmentation.gaussian_noise:
            pcd1 = add_noise(pcd1, sigma=self.cfg.lamda)
            pcd2 = add_noise(pcd2, sigma=self.cfg.lamda2)


        return pcd1.astype(np.float32), camera_matrix, pcd2.astype(np.float32), camera_matrix2, mesh_name,

    def __len__(self):
        return len(self.mesh_names)


class canonical_data_loader(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.catg = cfg.class_name
        self.cfg = cfg
        self.cat = []
        self.cat.append(NAMES2ID[cfg.class_name])

        annots = json.load(open(os.path.join(BASEDIR, cfg.data.annot_path)))
        annots = [annot for annot in annots if annot['class_id'] in self.cat]

        selected_cat = []
        for i in range(len(annots)):
            if annots[i]['class_id'] not in selected_cat:
                selected_cat.append(annots[i]['class_id'])
        print('loaded {} samples of categories: '.format(len(annots)), selected_cat)

        pcd_paths_np = []
        for i in range(len(selected_cat)):
            pcd_paths_np += glob(os.path.join(BASEDIR, cfg.data.pcd_root, selected_cat[i], '*.pcd'))

        # Calculate nclasses - handle case where semantic_id might not exist (self-supervised learning)
        try:
            self.nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1
        except KeyError:
            # For self-supervised learning, set nclasses to 1 since semantic classes are not used
            self.nclasses = 1

        split_models = open(os.path.join(BASEDIR, cfg.data.splits_root, "{}.txt".format(split))).readlines()
        split_models = [m.split('-', 1)[1].rstrip('\n') for m in split_models]

        mesh_names = []
        pointCloud_lst = []
        print("Loading {} data, please wait\n".format(split))
        for fn in tqdm(pcd_paths_np):
            model_id = os.path.basename(fn).split('.')[0]
            if model_id not in split_models:
                continue

            mesh_names.append(model_id)
            pointCloud_lst.append(naive_read_pcd(fn)[0])

        self.transformed_pcds = pointCloud_lst
        self.mesh_names = mesh_names
        print("\nmesh_names: {}".format(len(self.mesh_names)))
        print("\point clouds: {}".format(len(self.transformed_pcds)))


    def __getitem__(self, idx):
        pcd1 = self.transformed_pcds[idx]
        mesh_name = self.mesh_names[idx]

        if self.cfg.augmentation.normalize_pc:
            pcd1 = normalize_pc(pcd1)

        if self.cfg.augmentation.uniform_sampling:
            pcd1_updated = farthest_point_sample(pcd1, self.cfg.sample_points)

        else:
            pcd1_updated = pcd1

        if self.cfg.augmentation.gaussian_noise:
            pcd1_updated = add_noise(pcd1_updated, sigma=self.cfg.lamda)

        return pcd1_updated.astype(np.float32), mesh_name


    def __len__(self):
        return len(self.transformed_pcds)


class single_pcd_data_loader(torch.utils.data.Dataset):
    """
    Dataset that overfits on a single PCD by generating pairs via random rotations.

    Returns tuples compatible with training loss:
      (pc1[Nx3], rot1[3x3], pc2[Nx3], rot2[3x3], name)
    """
    def __init__(self, cfg, split: str):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.path = getattr(cfg.data, 'single_pcd_path', None)
        if self.path is None or len(str(self.path)) == 0:
            raise ValueError("cfg.data.single_pcd_path must be set for single_pcd_data_loader")

        p = os.path.join(BASEDIR, self.path) if not os.path.isabs(self.path) else self.path
        pc = o3d.io.read_point_cloud(p)
        if pc.is_empty():
            raise ValueError(f"Point cloud is empty: {p}")
        self.pc = np.asarray(pc.points, dtype=np.float32)
        self.name = os.path.splitext(os.path.basename(p))[0]

        # Optional chunking
        self.chunk_size = float(getattr(cfg.data, 'single_chunk_size', 0.0) or 0.0)
        self.chunk_overlap = float(getattr(cfg.data, 'single_chunk_overlap', 0.0) or 0.0)
        self.min_chunk_points = int(getattr(cfg.data, 'single_min_chunk_points', 0) or 0)

        self.chunk_indices = None
        if self.chunk_size > 0.0:
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(self.pc.astype(float))
            big = pc_o3d.get_axis_aligned_bounding_box()
            tiles = divide_bbox(big, side_length=self.chunk_size, overlap=self.chunk_overlap)
            inds = []
            for tile in tiles:
                mask = inside_aabb_mask(self.pc, tile)
                idxs = np.nonzero(mask)[0]
                if self.min_chunk_points > 0 and idxs.shape[0] < self.min_chunk_points:
                    continue
                if idxs.shape[0] > 0:
                    inds.append(idxs)
            if not inds:
                raise ValueError("No chunks meet the minimum point requirement; adjust single_chunk_size/overlap/min_chunk_points.")
            self.chunk_indices = inds

        # Fixed target points per sample (enforce for stable batching)
        self.target_points = int(getattr(cfg, 'sample_points', 2048) or 2048)

        # Length per epoch
        self.len = getattr(cfg.data, 'single_pairs_per_epoch_train', 1000) if split == 'train' \
            else getattr(cfg.data, 'single_pairs_per_epoch_val', 200)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Choose points (whole cloud or a chunk)
        if self.chunk_indices is not None:
            # round-robin over chunks for determinism
            cidx = idx % len(self.chunk_indices)
            pts = self.pc[self.chunk_indices[cidx]]
        else:
            pts = self.pc

        # Two random rotations
        R1 = _random_rotation_matrix().astype(np.float32)
        R2 = _random_rotation_matrix().astype(np.float32)

        pc1 = (pts @ R1.T).astype(np.float32)
        pc2 = (pts @ R2.T).astype(np.float32)

        if self.cfg.augmentation.normalize_pc:
            pc1 = normalize_pc(pc1)
            pc2 = normalize_pc(pc2)

        # Always enforce fixed-size sampling for batching
        pc1 = sample_fixed_points(pc1, self.target_points)
        pc2 = sample_fixed_points(pc2, self.target_points)

        if self.cfg.augmentation.gaussian_noise:
            pc1 = add_noise(pc1, sigma=self.cfg.lamda)
            pc2 = add_noise(pc2, sigma=self.cfg.lamda2)

        return pc1.astype(np.float32), R1, pc2.astype(np.float32), R2, self.name





def debug(data):
    '''

    Parameters
    ----------
    data :: loaded batch of [pc1, pose1, pc2, pose2, name]

    Returns :: visualize if the relative pose is correct or not
               1. Inverse transform of pc1 and pc2 should be in a same initial pose
               2. Transform(pose2,   Transform(Inv(pose1),pc1)) => pc1 should transform to the pose 2
    -------

    '''
    aa = data[0][0]
    bb = data[2][0]
    # taa = data[1][1][0][:, :3]
    # tbb = data[3][1][0][:, :3]
    taa = data[1][0]
    tbb = data[3][0]

    '''Transform both the PCs to original pose'''
    aa2 = torch.matmul(taa.double().T, aa.double().T).T
    bb2 = torch.matmul(tbb.double().T,bb.double().T).T

    pdb.set_trace()
    show_points(aa, bb, True)
    show_points(aa2,bb2, True)

    '''same as transformation 1 : separate transformations'''
    aa2bb = torch.matmul(tbb.double(), aa2.double().T).T
    show_points(aa2bb,bb, True)
    bb2aa = torch.matmul(taa.double(), bb2.double().T).T
    show_points(bb2aa,aa, True)

    '''same as transformation 2 : in one line'''
    aa3bb = torch.matmul(tbb.double() @ taa.double().T , aa.double().T).T
    show_points(aa3bb, bb, True)
    bb3aa = torch.matmul(taa.double() @ tbb.double().T, bb.double().T).T
    show_points(bb3aa, aa, True)

    ''' Batch wise transformation '''
    AA2BB = torch.transpose(torch.bmm( torch.bmm(data[3].double(), torch.transpose(data[1].double(),1,2)) , torch.transpose(data[0].double(),1,2)), 1,2)
    show_points(AA2BB[5], data[2][5], True)
    BB2AA = torch.transpose(torch.bmm(torch.bmm(data[1].double(), torch.transpose(data[3].double(), 1, 2)),torch.transpose(data[2].double(), 1, 2)), 1, 2)
    show_points(BB2AA[5], data[0][5], True)

def show_points(points1, points2=0, both=False):
    '''

    Parameters
    ----------
    points      point cloud  [2048, 3]
    kp          estimated key-points  [10, 3]
    both        if plot both or just the point clouds

    Returns     show the key-points/point cloud
    -------

    '''
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)

    if both == False:
        o3d.visualization.draw([pcd1])
    else:
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        o3d.visualization.draw([pcd1, pcd2])


# main to test dataloader pipeline
def test_imgs_loader(cfg):

    train_dataset = generic_data_loader(cfg, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                   num_workers=cfg.num_workers, drop_last=False)

    train_iter = tqdm(train_dataloader)
    for i, data in enumerate(train_iter):
        print(len(data))
        debug(data)
        pdb.set_trace()
        show_points(data[0][0])
        show_points(data[0][0], data[0][2], True)
        debug(data)
        plt.show()
        # functions_bank.show_keypoints(data[0][0], data[0][0])



@hydra.main(config_path='config', config_name='config')
def main(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    test_imgs_loader(cfg)

if __name__ == '__main__':
    main()
