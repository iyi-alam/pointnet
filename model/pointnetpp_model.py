import os
import sys
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(dir_path)

import torch
import torch.nn as nn
from pointnetpp_utils import PointSetAbstraction

class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.psa1 = PointSetAbstraction(npoints=512, radius=0.2, nsamples=32, 
                                        in_channels=in_channels, out_channels=[64, 64, 128], group_all=False)
        self.psa2 = PointSetAbstraction(npoints=128, radius=0.4, nsamples=64, 
                                        in_channels=128+3, out_channels=[128, 128, 256], group_all=False)
        self.psa3 = PointSetAbstraction(npoints=None, radius=None, nsamples=None,
                                        in_channels=256+3, out_channels=[256, 512, 1024], group_all=True)
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(p=0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(p=0.4),

            nn.Linear(256, num_classes)
        )
        
    def forward(self, points: torch.Tensor):
        
        # points, features = self.psa1(points, None)
        # points, features = self.psa2(points, features)
        # points, features = self.psa3(points, features)
        points, features = self.psa3(*self.psa2(*self.psa1(points, None)))
        features = features.squeeze(dim=1)
        return self.fc(features)

def accuracy(preds, target):
    preds = torch.argmax(preds, dim=1)
    correct = torch.sum(preds == target)
    return correct.item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random_pts = torch.rand(8,1000,3)
    features = torch.rand(8,1000,16)
    npoint = 10
    nsample = 8
    radius = 0.2

    model1 = PointSetAbstraction(npoints=100, nsamples=8, radius=0.2, in_channels=3, out_channels=[64, 64, 128], group_all=False)
    model2 = PointSetAbstraction(npoints=10, nsamples=8, radius=0.4, in_channels=128+3, out_channels=[128+3, 128+3, 512], group_all=False)
    model3 = PointSetAbstraction(npoints=None, nsamples=None, radius=None, in_channels=512+3, out_channels=[512, 1024], group_all=True)
    centroids1, new_features1 = model1(random_pts, features=None)
    centroids2, new_features2 = model2(centroids1, new_features1)
    centroids3, new_features3 = model3(centroids2, new_features2)
    print(centroids1.shape, new_features1.shape)
    print(centroids2.shape, new_features2.shape)
    print(centroids3.shape, new_features3.shape)

    pnet = PointNetPlusPlus(10, 3)
    preds = pnet(random_pts)
    print(preds.shape)