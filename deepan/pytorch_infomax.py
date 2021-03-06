import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax
import umap

from create_pyg_dataset import create_dataset, generate_masks
from create_table import create_binary_table
from utils import get_adjacency_matrix

dataset = 'NSCLC'
filepath_data = r'/media/administrator/INTERNAL3_6TB/TCGA_data/pyt_datasets/NSCLC/raw/numerical_data_308_2021-03-18.pt'

if dataset == 'Cora':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset)
    data_i = dataset[0]
else:
    # load pytorch data object
    data = torch.load(filepath_data)

    out_channels = data.num_nodes
    num_features = data.num_features
    data_i = generate_masks(data, 0.7, 0.2)

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder(data_i.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)
data = data_i.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(data.x, data.edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    z, _, _ = model(data.x, data.edge_index)
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask], max_iter=150)
    return acc


for epoch in range(1, 301):
    loss = train()
    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))

''' Analysis of network'''
with torch.no_grad():
    z, _, _ = model(data.x, data.edge_index)
    print(z)
    z_0 = z.cpu().numpy()
    projection = umap.UMAP(n_components=z_0, random_state=42)
    result = projection.fit_transform(z_0)
    result_df = pd.DataFrame({'firstdim': result[:, 0], 'seconddim': result[:, 1]})
    plt.scatter(result_df.iloc[:, 0], result_df.iloc[:, 1], s=20)
    plt.show()