import sys
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


def print_edge_pairs(edge_index):
    """
    Prints edge pairs from tensor object
    """
    np.set_printoptions(threshold=sys.maxsize)
    print(edge_index.numpy().T)


def draw_graph_inspect(graph=None, data=None):
    """
    Takes graph or pytorch data object and plots it inline based on NetworkX funtions
    """
    if graph is None:
        graph = to_networkx(data, to_undirected=True)
        count_edges =  data.edge_index.shape[1]
    else:
        count_edges = 0

    # draw graph in spring, random or circular formation
    print('Plotting graph structure')
    #nx.draw_random(graph, arrows=False, with_labels=False, node_size=100, linewidths=0.2, width=0.2)
    #nx.draw_circular(graph, arrows=False, with_labels=False, node_size=10, linewidths=0.2, width=0.2)
    nx.draw_spring(graph, arrows=False, with_labels=False, node_size=10, linewidths=0.2, width=0.2, label = f'Edges:{count_edges}')
    plt.show()

def get_adjacency_matrix(df=None, cutoff=0.5, metric='cosine'):
    """
    Patient nodes have an edge connecting them if their distance in the feature space is below a set threshold.
    takes binary matrix,calculates distance and cutoffs
    Returns: boolean patient-patient adjacency df
    Metric docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    """

    data = df.to_numpy()

    # generate distance matrix on cosine similarity / can also do euclid
    dist = cdist(data, data,
                 metric=metric)

    # cutoff subsetting
    closes = dist < cutoff  # also self links

    # create df so that names are present
    df_adj = pd.DataFrame(closes, index=df.index.values, columns=df.index.values)
    return df_adj


def to_graph(closes):
    # takes numpy array
    G = nx.from_numpy_matrix(closes.to_numpy())
    # nx.draw(G)
    return G


def plot_in_out_degree_distributions(edge_index, num_of_nodes, dataset_name):
    """
    Cloned from https://github.com/gordicaleksa/pytorch-GAT
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'

    # Store each node's input and output degree (they're the same for undirected graphs such as Cora)
    in_degrees = np.zeros(num_of_nodes, dtype=np.int)
    out_degrees = np.zeros(num_of_nodes, dtype=np.int)

    # Edge index shape = (2, E), the first row contains the source nodes, the second one target/sink nodes
    # Note on terminology: source nodes point to target/sink nodes
    num_of_edges = edge_index.shape[1]
    for cnt in range(num_of_edges):
        source_node_id = edge_index[0, cnt]
        target_node_id = edge_index[1, cnt]

        out_degrees[source_node_id] += 1  # source node points towards some other node -> increment its out degree
        in_degrees[target_node_id] += 1  # similarly here

    hist = np.zeros(np.max(out_degrees) + 1)
    for out_degree in out_degrees:
        hist[out_degree] += 1

    fig = plt.figure(figsize=(12, 8), dpi=100)  # otherwise plots are really small in Jupyter Notebook
    plt.plot(hist, color='blue')
    plt.xlabel('node degree')
    plt.ylabel('# nodes for a given out-degree')
    plt.title(f'Node out-degree distribution for {dataset_name} dataset')
    plt.xticks(np.arange(0, len(hist), 10.0))

    return fig

def plot_degree_hist(adj):
    """
    Takes an adjacency dataframe and plot a degree histogram
    Returns figure
    """
    aggre = np.sum(adj.values, axis=0)

    loners = aggre -1
    singles = np.count_nonzero(loners==0)
    maxims = np.max(loners)

    fig = plt.figure()
    plt.hist(aggre)
    plt.xlabel('node degree in bins')
    plt.ylabel('# nodes for a given out-degree')
    plt.title(f'Degree histogram with {singles}')

    return fig
