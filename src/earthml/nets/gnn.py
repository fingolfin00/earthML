import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import lightning as L
from ..lightning import EarthMLLightningModule

class GCN (EarthMLLightningModule):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(3, 4)  # First layer: 3 input features, 4 output features
        self.conv2 = GCNConv(4, 2)  # Second layer: 4 input features, 2 output features

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x