import torch
import torch.nn as nn

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k= k
        self.dummy = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.k, 64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.Conv1d(64, 128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(128, 1024,1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        
        self.linear_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, self.k * self.k),
        )
    
    def device(self):
        return self.dummy.device

    def forward(self, points: torch.tensor):
        """
        points are expected to be of shape Batch_Size x Num_Points x Point_Dimension (=k)
        """

        # Mini feature encoding to generate transformation matrix
        points = torch.transpose(points, 1, 2)
        points = self.conv_layers(points)
        points = nn.MaxPool1d(points.shape[-1])(points)
        points = torch.squeeze(points, dim=-1)
        points = self.linear_layers(points)

        # Add identity matrix
        idmat = torch.eye(self.k, dtype = torch.float32, device=self.device())
        idmat = idmat[None, :, :]

        return points.view(-1, self.k, self.k) + idmat

class PointNetEncoder(nn.Module):
    def __init__(self, 
                 input_tnf_size = 3,
                 feature_tnf_size = 64,
                 feature_size = 1024):
        
        super().__init__()
        self.in_transform = Tnet(k=input_tnf_size)
        self.feat_transform = Tnet(k=feature_tnf_size)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_tnf_size, feature_tnf_size, 1),
            nn.BatchNorm1d(feature_tnf_size),
            nn.ReLU(True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(feature_tnf_size, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(128, feature_size, 1),
            nn.BatchNorm1d(feature_size),
            nn.ReLU(True)
        )

    def forward(self, points):
        """
        points are expected to be of shape Batch_Size x Num_Points x Point_Dimension (=k)
        """
        # input transform with tnet and matrix multiplication
        in_tnf = self.in_transform(points)
        points = torch.bmm(points, in_tnf).transpose(1,2)

        # First shared MLP and then feature transform
        conv1_out = self.conv1(points).transpose(1,2)
        feat_tnf = self.feat_transform(conv1_out)
        points = torch.bmm(conv1_out, feat_tnf).transpose(1,2)

        # Last few shared MLPs operating on transformed features
        points = self.conv2(points)
        points = nn.MaxPool1d(points.shape[-1])(points).squeeze(dim=-1)
        return points, in_tnf, feat_tnf
        

class PointNet(nn.Module):
    def __init__(self, num_classes, dropout_probs = 0.3, k1=3, k2=64, feature_size=1024):
        super().__init__()
        self.encoder = PointNetEncoder(k1, k2, feature_size)
        self.linear_layers = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, 256),
            nn.Dropout(p=dropout_probs),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, num_classes)
        )
        #self.dropout = nn.Dropout(p=dropout_probs)

    def forward(self, points):
        """
        points are expected to be of shape Batch_Size x Num_Points x Point_Dimension (=k)
        """
        encoder_out, in_tnf, feat_tnf = self.encoder(points)
        return self.linear_layers(encoder_out), in_tnf, feat_tnf


class LossFunc(nn.Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
    
    def forward(self, preds, target, matA, matB):
        pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = torch.rand(8, 2048, 3).to(device)
    # tnet = Tnet(k=3).to(device)
    # print(tnet.device())

    # #points = torch.transpose(points, 1,2)
    # print("input shape: ", points.shape)
    # out = tnet(points)
    # print("output shape: ", out.shape)

    # tns = torch.bmm(points, out)
    # print(tns.shape)

    # encoder = PointNetEncoder(feature_size=1024).to(device)
    # out = encoder(points)
    # print(out.shape)

    model = PointNet(num_classes=10).to(device)
    out, matA, matB = model(points)
    print(out.shape, matA.shape, matB.shape)