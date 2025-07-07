import os
import numpy as np
import trimesh #type:ignore
import matplotlib.pyplot as plt
import open3d as o3d #type:ignore


def read_off(filepath: str) -> tuple[list, list]:
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if lines[0].strip() != 'OFF':
        raise ValueError("Not a valid OFF file. First line must be 'OFF'.")

    idx = 1
    while lines[idx].strip().startswith('#') or lines[idx].strip() == '':
        idx += 1

    # Read counts
    parts = lines[idx].strip().split()
    n_vertices, n_faces, _ = map(int, parts)
    idx += 1

    # Read vertices
    vertices = []
    for i in range(n_vertices):
        x, y, z = map(float, lines[idx + i].strip().split())
        vertices.append([x, y, z])

    idx += n_vertices

    # Read faces
    faces = []
    for i in range(n_faces):
        parts = list(map(int, lines[idx + i].strip().split()))
        face = parts[1:]  
        faces.append(face)

    return vertices, faces


def sample_points_from_off(file_path, n_points=1024):
    mesh = trimesh.load(file_path)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    
    # Normalize: center and scale to fit in unit sphere
    points -= points.mean(axis=0)  
    scale = np.max(np.linalg.norm(points, axis=1))  # farthest point from origin
    points /= scale
    
    return points  

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


def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/ModelNet10/ModelNet10'))
    subfolders = os.listdir(folder_path)
    object_list = []
    for elem in subfolders:
        p = os.path.join(folder_path, elem)
        if os.path.isdir(p):
            object_list.append(elem)
    
    
    # Select a random folder and go to its train directory
    rnd_object = np.random.choice(object_list)
    object_folder = os.path.join(folder_path, rnd_object, 'train')
    rnd_file = np.random.choice(os.listdir(object_folder))
    file_path = os.path.join(object_folder, rnd_file)
    print(file_path)

    # Read shape file
    vertices, faces = read_off(filepath=file_path)

    print(len(vertices))
    print(len(faces))

    # Sample points using trimesh
    points = sample_points_from_off(file_path=file_path, n_points=1024)
    print(points.shape)
    #visualize_point_cloud(points)
    plot_points(points, title=rnd_file)
    