import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def index_points_(points: torch.tensor, idx: torch.tensor):
    D = points.shape[-1]
    idx = idx.unsqueeze(-1).repeat(1,1,D)
    return torch.gather(points, dim=1, index=idx)

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sampling(points: torch.tensor, npoint: int) -> torch.tensor:
    """
    points: B,N,3 shaped point cloud co-ordinates
    npoints: number of points to be sampled from each point
    """
    device = points.device
    B, N, C = points.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def plot(points, centroids, indexed_points):
    points = points.squeeze(0)
    indexed_points = indexed_points.squeeze(0)
    centroids = centroids.squeeze(0)
    points = points.cpu().numpy()
    indexed_points = indexed_points.cpu().numpy()
    centroids = centroids.cpu().numpy()

    N = points.shape[0]
    mask = np.zeros(shape = (N,), dtype = np.int32)
    mask[centroids] = 1

    fig, ax = plt.subplots(1,2, figsize=(16,8))
    for i in range(N):
        color = "red" if mask[i] else "green"
        ax[0].scatter(points[i,0], points[i,1], c = color)
    
    for i in range(indexed_points.shape[0]):
        ax[1].scatter(indexed_points[i,0], indexed_points[i,1], c = "blue")
    
    plt.tight_layout()
    plt.show()


def plot_groups(grouped_pts):
    grouped_pts = grouped_pts.squeeze(0).cpu().numpy()
    S,K,C = grouped_pts.shape
    for g in range(S):
        for p in range(K):
            pt = grouped_pts[g][p]
            plt.scatter(pt[0], pt[1], color = "lightgray", s=10, alpha = 0.5)
            plt.text(pt[0], pt[1], s = f"{g}", fontsize=8, ha="center", va="center")
    
    plt.show()

def sample_and_group(xyz: torch.Tensor, features: torch.Tensor,
                     npoints, nsamples, radius) -> tuple[torch.Tensor, torch.Tensor]:
    """
    xyz: co-ordinates of points of shape B x N x 3
    features: features of each point of shape B x N x D
    """
    B,N,C = xyz.shape
    centroids_idx = farthest_point_sampling(xyz, npoints)
    centroids = index_points(xyz, centroids_idx)

    grouping_idx = query_ball_point(radius, nsamples, xyz, centroids)
    grouped_xyz = index_points(xyz, grouping_idx)
    normalized_grouped_xyz = grouped_xyz - centroids.view(B, npoints, 1, C)

    if features is not None:
        # concatenate features along with point co-ordinates
        grouped_features = index_points(features, grouping_idx)
        new_features = torch.cat((normalized_grouped_xyz, grouped_features), dim=-1)
    else:
        new_features = grouped_xyz
    
    return centroids, new_features


def sample_and_group_all(xyz: torch.Tensor, features: torch.Tensor):
    """
    xyz: co-ordinates of points of shape B x N x 3
    features: features of each point of shape B x N x D
    """
    centroids = torch.mean(xyz, dim=1, keepdim=True)
    if features is not None:
        new_features = torch.cat((xyz, features), dim = -1).unsqueeze(dim=1)
    else:
        new_features = xyz.unsqueeze(dim=1)
    return centroids, new_features


class PointSetAbstraction(nn.Module):
    def __init__(self, npoints, nsamples, radius, in_channels, out_channels, group_all = False):
        super().__init__()
        if not group_all:
            self.sampler = partial(sample_and_group, npoints=npoints, nsamples=nsamples, radius=radius)
        else:
            self.sampler = sample_and_group_all
        layers = []

        prev_channels = in_channels
        for out_channel in out_channels:
            layers += [
                nn.Conv2d(prev_channels, out_channel, 1,1,0),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
            ]
            prev_channels = out_channel
        
        self.module = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        """
        xyz: co-ordinates of points of shape B x N x 3
        features: features of each point of shape B x N x D
        """
        centroids, new_features = self.sampler(xyz=xyz, features=features)
        new_features = new_features.permute(0,3,1,2)
        new_features = self.module(new_features)
        new_features = torch.max(new_features, dim=-1)[0]
        new_features = new_features.permute(0,2,1)
        return centroids, new_features




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random_pts = torch.rand(1,1000,3)
    features = torch.rand(1,1000,16)
    npoint = 10
    nsample = 8
    radius = 0.2
    # centroids = farthest_point_sampling(random_pts, npoint)
    # print(centroids.shape)

    
    # indexed_pts = index_points(random_pts, centroids)
    # print(indexed_pts.shape)

    # #plot(random_pts, centroids, indexed_pts)

    # #Another experiment
    # group_idx = query_ball_point(2, nsample=8, xyz=random_pts, new_xyz=indexed_pts)
    # print(group_idx.shape)

    # grouped_pts = index_points(points=random_pts, idx=group_idx)
    # print(grouped_pts.shape)
    # plot_groups(grouped_pts=grouped_pts)
    # centroids, new_features = sample_and_group(random_pts, features=features, npoints=npoint, nsamples=nsample, radius=radius)
    # print(centroids.shape, new_features.shape)

    # centroids, new_features = sample_and_group_all(random_pts, features=features)
    # print(centroids.shape, new_features.shape)

    model1 = PointSetAbstraction(npoints=100, nsamples=8, radius=0.2, in_channels=3, out_channels=[64, 64, 128], group_all=False)
    model2 = PointSetAbstraction(npoints=10, nsamples=8, radius=0.4, in_channels=128+3, out_channels=[128+3, 128+3, 512], group_all=False)
    model3 = PointSetAbstraction(npoints=None, nsamples=None, radius=None, in_channels=512+3, out_channels=[512, 1024], group_all=True)
    centroids1, new_features1 = model1(random_pts, features=None)
    centroids2, new_features2 = model2(centroids1, new_features1)
    centroids3, new_features3 = model3(centroids2, new_features2)
    print(centroids1.shape, new_features1.shape)
    print(centroids2.shape, new_features2.shape)
    print(centroids3.shape, new_features3.shape)




    