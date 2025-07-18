import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import random

def plot3D(file_path: str, title: str, savedir = None):
    # Load the mesh
    mesh = trimesh.load(file_path)
    object_name = file_path.split("/")[-1][:-4]

    # Get vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    # Create figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh using a Poly3DCollection
    mesh_collection = Poly3DCollection(vertices[faces], alpha=0.7, edgecolor='k')
    mesh_collection.set_facecolor((0.5, 0.5, 1, 0.6))  # light blue

    ax.add_collection3d(mesh_collection)

    # Auto scale to the mesh size
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # Add labels and title
    ax.set_title(object_name, fontsize=16)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Remove grid for cleaner view
    ax.grid(False)

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1,1,1]) 

    plt.tight_layout()
    if savedir:
        save_path = os.path.join(savedir, f"modelnet_{object_name}.png")
        plt.savefig(save_path)
    else:
        plt.show()

def plot_3d_grid(file_paths, titles = None, savedir=None, plot_name = None):
    assert len(file_paths) == 8, "Expected exactly 8 file paths"

    fig = plt.figure(figsize=(16, 12))

    for idx, file_path in enumerate(file_paths):
        mesh = trimesh.load(file_path)
        object_name = os.path.basename(file_path)[:-4] if not titles else titles[idx]

        vertices = mesh.vertices
        faces = mesh.faces

        ax = fig.add_subplot(4, 2, idx + 1, projection='3d')

        mesh_collection = Poly3DCollection(vertices[faces], alpha=0.7, edgecolor='k')
        mesh_collection.set_facecolor((0.5, 0.5, 1, 0.6))
        ax.add_collection3d(mesh_collection)

        scale = vertices.flatten()
        ax.auto_scale_xyz(scale, scale, scale)

        ax.set_title(object_name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    if savedir:
        os.makedirs(savedir, exist_ok=True)
        save_path = os.path.join(savedir, f"{plot_name}.png" if plot_name else "modelnet_grid.png")
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    
    dataroot = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/ModelNet10/ModelNet10"))
    savedir = os.path.join(os.path.dirname(__file__), "outputs")
    folder_list = os.listdir(dataroot)

    N = 8
    file_paths = []
    for _ in range(N):
        select_folder = random.choice(folder_list)
        folder_path = os.path.join(dataroot, select_folder)
        while (not os.path.isdir(folder_path)):
            select_folder = random.choice(folder_list)
            folder_path = os.path.join(dataroot, select_folder)
        file_list = os.listdir(os.path.join(folder_path, "test"))
        select_file = random.choice(file_list)
        while (not select_file.__contains__(".off")):
            select_file = random.choice(file_list)
        file_paths.append(os.path.join(folder_path,"test" ,select_file))
    
    #print(file_paths)
    plot_3d_grid(file_paths, savedir)

