import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import trimesh  #type:ignore
import open3d as o3d #type:ignore
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


class ModelNet(Dataset):
    def __init__(self, dataroot, transform, n_points=1024, dset = "train"):
        self.dataroot = dataroot
        self.transform = transform
        self.dset = dset
        self.n_points = n_points
        self.file_paths, self.obj_to_key, self.key_to_obj = self.prepare_file_list()

    
    def prepare_file_list(self):
        file_paths = []
        subfolders = os.listdir(self.dataroot)
        object_list = []
        for elem in subfolders:
            p = os.path.join(self.dataroot, elem, self.dset)
            if os.path.isdir(p):
                object_list.append(elem)
                for filename in os.listdir(p):
                    file_paths.append(
                        os.path.join(p, filename)
                    )

        np.random.shuffle(file_paths)
        obj_to_key = {obj_name: label for label, obj_name in enumerate(object_list)}
        key_to_obj = {label: obj_name for label, obj_name in enumerate(object_list)}
        return file_paths, obj_to_key, key_to_obj
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        obj_name = file_path.split("/")[-3]
        label = self.obj_to_key[obj_name]
        points = self.sample_points_from_off(file_path=file_path, n_points=self.n_points)
        if self.transform:
            points = self.transform(points)
        
        return points, torch.tensor(label)
    
    def sample_points_from_off(self, file_path, n_points=1024):
        mesh = trimesh.load(file_path)
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
        return points


class Augmentation:
    def __init__(self, max_rotate, pos_noise_std):
        self.max_rotate = max_rotate
        self.pos_noise_std = pos_noise_std
    
    def translate(self, points: np.ndarray) -> np.ndarray:
        noise = np.random.normal(loc=0, scale=self.pos_noise_std, size=points.shape)
        return points + noise
    
    def rotate(self, points: np.ndarray) -> np.ndarray:
        axis = np.array([0.0,0.0,1.0])
        theta = np.random.uniform(low=-self.max_rotate, high=self.max_rotate) * np.pi/180
        rot_vec = theta * axis
        rotation = R.from_rotvec(rot_vec)
        return rotation.apply(points)

    def normalize(self, points: np.ndarray) -> np.ndarray:
        # Normalize: center and scale to fit in unit sphere
        points -= points.mean(axis=0)  
        scale = np.max(np.linalg.norm(points, axis=1))  # farthest point from origin
        points /= scale
        return points 


def get_train_transform():
    aug = Augmentation(max_rotate=10, pos_noise_std=0.02)
    return transforms.Compose([
        aug.normalize,
        aug.rotate,
        aug.translate,
        transforms.ToTensor()
    ])
    

def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def plot_points(points: np.ndarray, title):
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    ax.scatter(x, y, z, c=z, cmap='viridis', s=1)  
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=-50)  
    plt.tight_layout()
    plt.show()

        
if __name__ == "__main__":
    dataroot = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/ModelNet10/ModelNet10"))
    train_transform = get_train_transform()
    
    modelnet = ModelNet(dataroot=dataroot, transform=train_transform)
    print(len(modelnet.file_paths))
    print(modelnet.obj_to_key)
    point, label = modelnet.__getitem__(10)
    print(point.shape, label)
    point = point.numpy()
    plot_points(point[0], title = f"{modelnet.key_to_obj[label.item()]}")
