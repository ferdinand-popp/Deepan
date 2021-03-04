import argparse
import os
import os.path as osp
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.manifold import TSNE, MDS
import umap
from sklearn.metrics import silhouette_score
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
from survival_analysis import create_survival_plot


def get_arguments():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default='NSCLC',
                        choices=['Cora', 'CiteSeer', 'PubMed', 'LUAD', 'NSCLC'])
    parser.add_argument('--newdataset', action='store_true', default='True')
    parser.add_argument('--cutoff', type=float, default=0.5)
    parser.add_argument('--filepath_dataset',
                        default=r'/media/administrator/INTERNAL3_6TB/TCGA_data/pyt_datasets/LUAD/raw/data_208_2021-02-24.pt')
    # Model
    parser.add_argument('--variational', default='False')
    parser.add_argument('--linear', default='True')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.005)
    # parser.add_argument('--decay', type=float, default=0.6)
    parser.add_argument('--outputchannels', type=int, default=208)
    # Embedding
    parser.add_argument('--projection', type=str, default='UMAP',
                        choices=['TSNE', 'UMAP', 'MDS', 'LEIDEN'])
    # parser.add_argument('--visualize', action='store_true', default='False')
    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    # parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging", default=False)
    args = parser.parse_args()
    return args


args = get_arguments()

'''Dataset selection and generation'''
# added line for our LUAD set
if args.dataset in ['LUAD', 'NSCLC']:
    if args.newdataset == 'True':
        # read basic data and preselect it into dfs
        df_features, df_y = create_binary_table(dataset=args.dataset, clinical=True, mutation=True, expression=True)
        # calculate adjacency matrix based on feature distance
        df_adj = get_adjacency_matrix(df=df_features, cutoff=args.cutoff, metric='cosine')
        # use generated dfs to save as pytorch data object
        dataset_unused, filepath = create_dataset(datasetname=args.dataset, df_adj=df_adj, df_features=df_features,
                                                  df_y=df_y)  # contains .survival redundant
    else:
        # !!! use existing data object
        filepath = args.filepath_dataset

    # load data
    data = torch.load(filepath)
    df_y = data.survival
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


'''
class MarginalizedLinearDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MarginalizedLinearDecoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, cached=True)
    
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        value = torch.sigmoid(value) if sigmoid else value
        
        x_recon = self.conv(x, edge_index)
        
        return value, x_recon 
        '''

'''Selection of Model'''
if args.variational == 'False':
    if args.linear == 'False':
        model = GAE(GCNEncoder(num_features, out_channels))
        model_name = 'GCN'
    else:
        if args.linear == 'MGAE':
            # model = GAE(encoder=LinearEncoder(num_features, out_channels), decoder=MarginalizedLinearDecoder(out_channels, num_features))
            model_name = 'MGAE'
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
writer_folder = f'{len(dirs)}-{model_name}'  # number of folder +1 naming
path_complete = os.path.join(logpath, writer_folder)
writer = SummaryWriter(log_dir='../runs/{}/{}'.format(date.today(), writer_folder))

'''GPU CUDA Connection and send data'''
# seeding
torch.manual_seed(0)  # np.random.seed(0) # torch.set_deterministic(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # add decay?


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    if args.linear == 'MGAE':
        loss = model.recon_loss(z, data.edge_index, feature_matrix=data.x)
    else:
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
    return model.test(z, pos_edge_index, neg_edge_index)


def projection(z, dimensions=2):
    print('Projection')
    if args.projection == 'TSNE':
        projection = TSNE(n_components=dimensions, random_state=123)
    elif args.projection == 'UMAP':
        projection = umap.UMAP(n_components=dimensions, random_state=42)  # n_neighbors=30
    elif args.projection == 'MDS':
        projection = MDS(n_components=dimensions)
    elif args.projection == 'LEIDEN':
        # see https://github.com/vtraag/leidenalg IGraph
        pass
    # for graphs?
    else:
        print('No projection')
        pass
    result = projection.fit_transform(z)
    # TODO add all dimens into frame
    result_df = pd.DataFrame({'firstdim': result[:, 0], 'seconddim': result[:, 1]})
    return result_df


def plot_silhoutte_comparison(result_df):
    n_clusters = len(set(result_df.labels))
    X = result_df.iloc[:, [0, 1]].to_numpy()
    y = result_df.labels.to_numpy()
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, y)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, y)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[y == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    '''
    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    '''

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()
    return fig


def clustering_points(result_df):
    # Clustering input result df
    clustering = DBSCAN(min_samples=2).fit(result_df.to_numpy())
    labels = clustering.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(f'DBSCAN: Clusters: {n_clusters_}, Excluded points:{n_noise_}')

    if n_clusters_ > 1:
        data.silhoutte_score = silhouette_score(result_df, labels)

        # PLot silhoutte
        fig_silhoutte = plt.figure(figsize=(8, 8))
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors_ = [plt.cm.Spectral(each)
                   for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors_):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[clustering.core_sample_indices_] = True
            xy = result_df[class_member_mask & core_samples_mask]
            plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = result_df[class_member_mask & ~core_samples_mask]
            plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

        plt.title('DBSCAN number of clusters: %d' % len(unique_labels))
        # plt.show()
        writer.add_figure('Clustering', fig_silhoutte, epoch)

    else:
        data.silhoutte_score = 0
    return labels


def cluster_patients(df_y):
    with torch.no_grad():
        # get representation (nodes, outputchannels(feature dimensions))
        z = model.encode(x, train_pos_edge_index)
        z_0 = z.cpu().numpy()  # copies it to CPU

        # transform presentation like in DEEPAN
        z_1 = np.dot(z_0, z_0.T)  # inner dot product (nodes, nodes) returned
        z_2 = (np.absolute(z_1) + np.absolute(z_1.T)) / 2
        # symmetric and nonnegative representation (nodes, nodes) returned

        # Embedding projection
        result_df = projection(z_0)  # UMAP, TSNE, LEIDEN, MDS

        labels = clustering_points(result_df)
        result_df['labels'] = labels  # have same order for nodes
        df_y['labels'] = labels  # have same order for nodes

        # Plot clustering
        plot_embedding(result_df)

        # coeffcients
        #figure = plot_silhoutte_comparison(result_df)
        #writer.add_figure('Silhoutte coeffcient', figure, epoch)

        # Make Df ready for survival analysis
        df_y.rename(columns={'OS_time_days': 'days_to_death', 'OS_event': 'vital_status'}, inplace=True)
        df_y.index.name = None
        df_y.to_csv(os.path.join(path_complete, 'df_y.csv'), index=True, sep="\t")

        figure_survival = create_survival_plot(df_y)
        writer.add_figure('Survival', figure_survival)
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
        for i in df.labels.unique():
            df_i = df.loc[df['labels'] == i]
            plt.scatter(df_i.iloc[:, 0], df_i.iloc[:, 1], s=20, color=colors[i + 2])
    else:
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=20)

    title = '{}, Model: {}, Features: {}, AUC: {}, Silhoutte Score:{}'.format(args.projection, model_name,
                                                                              data.num_features, round(best_val_auc, 3),
                                                                              data.silhoutte_score)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # plt.show()
    writer.add_figure('Projection', fig, epoch)


# Train Network
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

# Call projection and clustering and plotting
cluster_patients(df_y)  # writes also

writer.close()

# save model
# modelpath = os.path.join(os.getwd(), 'models')
# torch.save(model.state_dict(), modelpath)
