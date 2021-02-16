import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges

from create_pyg_dataset import create_dataset, generate_masks
from create_table import create_binary_table
from utils import get_adjacency_matrix, plot_in_out_degree_distributions

# seeding
torch.manual_seed(0)  # np.random.seed(0) # torch.set_deterministic(True)

parser = argparse.ArgumentParser()
parser.add_argument('--variational', default='True')
parser.add_argument('--linear', default='False')
parser.add_argument('--dataset', type=str, default='LUAD',
                    choices=['Cora', 'CiteSeer', 'PubMed', 'LUAD'])
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--cutoff', type=float, default=0.4)
parser.add_argument('--visualize', action='store_true', default='False')
parser.add_argument('--newdataset', action='store_true', default='True')
parser.add_argument('--decay', type=float, default=0.6)
parser.add_argument('--outputchannels', type=int, default=5)

# Logging/debugging/checkpoint related (helps a lot with experimentation)
# parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging", default=False)

args = parser.parse_args()

# Writer will output to ./runs/ directory by default
_, dirs, _ = next(os.walk(r'/home/fpopp/PycharmProjects/Deepan/runs/160221'))  # get folder in runs
writer_folder = f'{len(dirs)}'  # number of folder +1 naming
writer = SummaryWriter(r'../runs/{}'.format(writer_folder), comment='Help', filename_suffix='Yes')

# added line for our LUAD set
if args.dataset == 'LUAD':
    if args.newdataset == 'True':
        # 1
        df_features, df_y = create_binary_table(clinical=True, mutation=True, expression=True)
        # 2
        df_adj = get_adjacency_matrix(df_features, cutoff=args.cutoff, metric='cosine')
        # 3
        dataset_unused, filepath = create_dataset(df_adj, df_features, df_y)  # contains .survival redundant
    else:  # use existing data obejct
        filepath = r'/media/administrator/INTERNAL3_6TB/TCGA_data/LUAD/raw/data_208_2021-02-11.pt'
    data = torch.load(filepath)
    data = generate_masks(data, 0.7, 0.2)
else:
    # Planetoid dataset
    path_data = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    datasets = Planetoid(path_data, args.dataset, transform=T.NormalizeFeatures())
    data = datasets[0]

out_channels = args.outputchannels
num_features = data.num_features
#plot_in_out_degree_distributions(data.edge_index, data.num_nodes, args.dataset)
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)

'''Models'''


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


'''Selection'''

if not args.variational:
    if not args.linear:
        model = GAE(GCNEncoder(num_features, out_channels))
        model_name = 'Graph Autoencoder GCN'
    else:
        model = GAE(LinearEncoder(num_features, out_channels))
        model_name = 'Linear encoder'

else:
    if args.linear:
        model = VGAE(VariationalLinearEncoder(num_features, out_channels))
        model_name = 'Variantional Linear Encoder'

    else:
        model = VGAE(VariationalGCNEncoder(num_features, out_channels))
        model_name = 'Graph Variational Autoencoder GCN'

'''GPU CUDA Connection and send data'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # add decay?


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    if args.variational:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
        meb = z.cpu().numpy()  # return nxn numpy array
        '''
        # Cluster embedded values using k-means.
        kmeans_input = z.cpu().numpy() #copies it to CPU 
        kmeans = KMeans(n_clusters=args.clusters, random_state=0).fit(kmeans_input)
        kmeans.labels # e.g: array([1, 1, 1, 0, 0, 0], dtype=int32)
        '''
    return model.test(z, pos_edge_index, neg_edge_index)

#logging
params = ''
for arg in vars(args):
    params += '{}: {} \n'.format(arg, getattr(args, arg) or '')
writer.add_text('Parameters', params)
writer.add_hparams(vars(args), {'linear': 0})


losses = []
aucs = []
for epoch in range(1, args.epochs + 1):
    loss = train()
    losses.append(loss)
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    writer.add_scalar('auc', auc, epoch)
    writer.add_scalar('loss', loss, epoch)

    aucs.append(auc)
    print('Epoch: {:03d}, Loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))

writer.add_scalar('Top', auc)

def plot_auc(aucs):
    # plt.plot(losses)
    plt.plot(aucs)
    title = '{}, Model: {}, Features: {}, AUC: {}'.format(args.dataset, model_name, data.num_features, max(aucs))
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    # plt.show()

    print('Done')


'''
@torch.no_grad()
def plot_points(colors):
    model.eval()
    z = model.encode(data.x, data.train_pos_edge_index)
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    # y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        # plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
        plt.scatter(z[i, 0], z[i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.show()

colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
]

plot_points(colors)
'''


def cluster_patients():
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
        # Cluster embedded values using k-means.
        kmeans_input = z.cpu().numpy()  # copies it to CPU
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(kmeans_input)
            writer.add_scalar('SSE', kmeans.inertia_, k)

        #categories = kmeans.labels_  # e.g: array([1, 1, 1, 0, 0, 0], dtype=int32)
        #df_y['category'] = pd.Series(categories)

cluster_patients()
