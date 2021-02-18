import argparse
import os
import os.path as osp
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges
from scipy.spatial.distance import cdist

from create_pyg_dataset import create_dataset, generate_masks
from create_table import create_binary_table
from utils import get_adjacency_matrix, plot_in_out_degree_distributions

# seeding
torch.manual_seed(0)  # np.random.seed(0) # torch.set_deterministic(True)

parser = argparse.ArgumentParser()
parser.add_argument('--variational', default='False')
parser.add_argument('--linear', default='True')
parser.add_argument('--dataset', type=str, default='LUAD',
                    choices=['Cora', 'CiteSeer', 'PubMed', 'LUAD'])
parser.add_argument('--projection', type=str, default='UMAP',
                    choices=['TSNE', 'UMAP'])
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--cutoff', type=float, default=0.5)
parser.add_argument('--visualize', action='store_true', default='False')
parser.add_argument('--newdataset', action='store_true', default='False')
parser.add_argument('--decay', type=float, default=0.6)
parser.add_argument('--outputchannels', type=int, default=208)
# Logging/debugging/checkpoint related (helps a lot with experimentation)
# parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging", default=False)
args = parser.parse_args()

'''Dataset selection and generation'''
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
    # data = generate_masks(data, 0.85, 0.1) #node masks for dataset train and test values and rest val
else:
    # Planetoid dataset
    path_data = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    datasets = Planetoid(path_data, args.dataset, transform=T.NormalizeFeatures())
    data = datasets[0]

num_features = data.num_features
out_channels = args.outputchannels
# plot_in_out_degree_distributions(data.edge_index, data.num_nodes, args.dataset)
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
if args.variational == 'False':
    if args.linear == 'False':
        model = GAE(GCNEncoder(num_features, out_channels))
        model_name = 'GCN'
    else:
        model = GAE(LinearEncoder(num_features, out_channels))
        model_name = 'Linear'
else:
    if args.linear == 'True':
        model = VGAE(VariationalLinearEncoder(num_features, out_channels))
        model_name = 'VarLinear'

    else:
        model = VGAE(VariationalGCNEncoder(num_features, out_channels))
        model_name = 'VarGCN'

'''Logging'''
# Writer will output to ./runs/ directory by default
logpath = os.path.join(os.path.split(os.getcwd())[0], f'runs/{date.today()}')
if not os.path.exists(logpath):
    os.makedirs(logpath)
_, dirs, _ = next(os.walk(logpath))  # get folder in logpath
writer_folder = f'{len(dirs)} - {model_name}'  # number of folder +1 naming
writer = SummaryWriter(log_dir='../runs/{}/{}'.format(date.today(), writer_folder))

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
    if args.variational == 'True':
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


def cluster_patients():
    with torch.no_grad():
        # get representation (nodes, outputchannels(feature dimensions))
        z = model.encode(x, train_pos_edge_index)
        # Cluster embedded values using k-means.
        z_0 = z.cpu().numpy()  # copies it to CPU

        z_1 = np.dot(z_0, z_0.T)  # inner dot product (nodes, nodes) returned
        z_2 = (np.absolute(z_1) + np.absolute(z_1.T)) / 2
        # symmetric and nonnegative representation (nodes, nodes) returned

        # PCA KMeans
        # DBSCAN

        # Visualizing
        print('Projection')
        if args.projection == 'TSNE':
            projection = TSNE(n_components=2, random_state=123)
        else:
            projection = umap.UMAP(n_neighbors=30,
                                   n_components=2,
                                   random_state=42)  # more params
        result = projection.fit_transform(z_2)
        result_df = pd.DataFrame({'firstdim': result[:, 0], 'seconddim': result[:, 1]})

        plot_embedding(result_df)

        clustering = DBSCAN(eps=3, min_samples=2).fit(z_0)
        result_df['labels']=clustering.labels_

        plot_embedding(result_df)
        '''New clusters were identified using a spectral clustering algorithm, 
        which was done by running kmeans on the top number of clusters eigenvectors 
        of the normalized Laplacian z_2  
        #eigenvalues, eigenvectors = np.linalg.eig(z_2)
        
                for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(kmeans_input)
            # Docu Elbow Inertia
            writer.add_scalar('SSE', kmeans.inertia_, k)

            # Docu Elbow Distortion
            distortion = sum(np.min(cdist(kmeans_input, kmeans.cluster_centers_, 'euclidean'), axis=1)) / \
                         kmeans_input.shape[0]
            writer.add_scalar('Distortion', distortion, k)

        # embedding with categories as colors and tSNE
        # writer.add_embedding(kmeans_input, metadata=categories) # not executable on firefox due to WebGL acceleration
         '''


colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700', '#092345', '#ffc456', '#69b1b3',
    '#76c474', '#ebdf6a'
]


def plot_embedding(df):
    fig = plt.figure(figsize=(8, 8))
    if len(df.columns) > 2:  # contains labels?
        for i in range(0, df.labels.unique().max() + 1):
            df_i = df.loc[df['label'] == i]
            plt.scatter(df_i.iloc[:, 0], df_i.iloc[:, 1], s=20, color=colors[i])
    else:
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=20)
    title = '{}, Model: {}, Features: {}, AUC: {}'.format(args.projection, model_name, data.num_features, best_val_auc)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # plt.show()
    writer.add_figure('Projection', fig, epoch)


best_val_auc = 0
for epoch in range(1, args.epochs + 1):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    if auc > best_val_auc:
        best_val_auc = auc
    writer.add_scalar('auc', auc, epoch)
    writer.add_scalar('loss', loss, epoch)

    print('Epoch: {:03d}, Loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))


def plot_auc(aucs):
    # plt.plot(losses)
    plt.plot(aucs)
    title = '{}, Model: {}, Features: {}, AUC: {}'.format(args.dataset, model_name, data.num_features, max(aucs))
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    # plt.show()

    print('Done')


'''Logging Parameters'''
param_dict = vars(args)
param_dict['Model'] = model_name
params = ''
for arg in param_dict:
    params += '{}: {} \n'.format(arg, getattr(args, arg) or '')
writer.add_text('Parameters', params)
writer.add_hparams(param_dict, {'AUC_best': best_val_auc})
writer.add_scalar('AUC_best', best_val_auc)

cluster_patients()  # writes also

writer.close()

# save model
# modelpath = os.path.join(os.getwd(), 'models')
# torch.save(model.state_dict(), modelpath)
