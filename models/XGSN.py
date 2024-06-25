import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.neighbors import KDTree
import dgl.function as fn
import dgl.nn as dglnn


class DGCNN(nn.Module):
    def __init__(self):
        super(DGCNN, self).__init__()
        self.num_classes = 128

        # local feature extraction module
        self.local = LocalOperation()

        # graph feature extraction module
        self.graph = GraphConvolution()

        # global feature extraction module
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 128, bias=True)

    def forward(self, pointcloud):
        # Local Operation
        x = self.local(pointcloud)

        # Construct knn graph
        edge_index = knn(pointcloud)

        # Graph ConvNet Layers
        x, edge_index = self.graph(x, edge_index)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        # Global feature extraction
        x_pool = torch.max(x, dim=0)[0]
        x = self.fc3(x_pool)

        return x


class LocalOperation(nn.Module):
    def __init__(self):
        super(LocalOperation, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

    def forward(self, x):
        batchsize = x.size()[0]

        # Sort point clouds by distance to their centroid
        centroids = torch.mean(x, dim=2)
        dists = ((x - centroids.view(batchsize, -1, 1, 1)) ** 2).sum(dim=1)
        idx = dists.argsort(dim=-1)
        x = x.transpose(2, 3)
        x = torch.gather(x, dim=-1, index=idx.expand(-1, -1, -1, x.size(-1)))

        # Local ConvNet
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.max(x, dim=-1)[0]

        return x


class GraphConvolution(nn.Module):
    def __init__(self):
        super(GraphConvolution, self).__init__()
        self.conv1 = dglnn.GraphConv(128, 128)
        self.conv2 = dglnn.GraphConv(128, 128)
        self.conv3 = dglnn.GraphConv(256, 256)

    def forward(self, x, edge_index):
        # graph convnet layer 1
        x = F.relu(self.conv1(x, edge_index))

        # graph convnet layer 2
        x = F.relu(self.conv2(x, edge_index))

        # graph convnet layer 3
        x = self.conv3(x, edge_index)

        return x, edge_index


# 定义多层MLP模型
class PointCloudMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PointCloudMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),  # 输入特征维度为1539，输出特征维度为512
            nn.ReLU(),
            nn.Linear(512, 1024),  # 输入特征维度为512，输出特征维度为1024
            nn.ReLU(),
            nn.Linear(1024, output_dim)  # 输入特征维度为1024，输出特征维度为2048*3
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(-1, 3, 245000)  # 将输出展平为形状为 (60, 3, 2048) 的点云
        return x


def knn(point_cloud):
    # 调整 point_cloud 的形状为 (batch_size * num_points, 3)
    batch_size, _, num_points, _ = point_cloud.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = point_cloud.view(batch_size * num_points, 3)
    # 构建 KD-Tree
    k = 5  # 假设 K 为每个点的最近邻数目
    kdtree = KDTree(x.cpu().numpy())

    # 查询每个点的最近邻
    distances, indices = kdtree.query(x.cpu().numpy(), k=k)

    # 将 indices 转换为边连接关系 edge_index
    # 构建边连接关系时，需要考虑 batch 中的索引
    # 我们将每个 batch 中的点的索引进行偏移，以便它们在整个点云中保持唯一
    offsets = torch.arange(0, batch_size * num_points, step=num_points, dtype=torch.long, device=x.device)
    offsets = offsets.unsqueeze(1).unsqueeze(1).expand(batch_size, num_points, k)
    offsets = offsets.reshape(batch_size * num_points, k)
    edge_index = torch.tensor(indices, device=device) + offsets
    edge_index = edge_index.view(2, -1)

    return edge_index


class MVPCG(nn.Module):
    def __init__(self):
        super(MVPCG, self).__init__()

        self.dgcnn = DGCNN()

        self.img_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.img_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.img_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.mlp = PointCloudMLP(128 + 3 * 256, 245000 * 3)

    def forward(self, pc, mv):
        batch_size, input_dim, num_points, _ = pc.size()
        # output = F.interpolate(pc, size=(5000, 1), mode='bilinear', align_corners=True)
        # # 提取点云的特征
        # edge_index = knn(output)
        # pc = output.view(batch_size * 5000, 3)
        pc = self.dgcnn(pc)

        # extract features from input images
        img_features = []
        for i in range(mv.size(1)):
            x = F.relu(self.img_conv1(mv[:, i]))
            x = F.relu(self.img_conv2(x))
            x = F.relu(self.img_conv3(x))
            x = F.max_pool2d(x, kernel_size=x.size()[2:])
            img_features.append(x.view(batch_size, -1))

        # concatenate point cloud and image features
        x = torch.cat(pc + img_features, dim=1)

        # fully connected layers for final fusion
        x = self.mlp(x)
        return x
