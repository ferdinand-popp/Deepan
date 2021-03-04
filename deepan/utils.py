import sys
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt


def print_edge_pairs(edge_index):
    np.set_printoptions(threshold=sys.maxsize)
    print(edge_index.numpy().T)


def get_adjacency_matrix(df=None, cutoff=0.5, metric='cosine'):
    # takes binary matrix,calculates distance and cutoffs -> returns boolean distance df
    if df is None:
        df = pd.read_csv(r'/media/administrator/INTERNAL3_6TB/TCGA_data/all_binary_selected.txt', index_col=0, sep='\t')

    data = df.to_numpy()
    n, m = data.shape

    # generate distance matrix on cosine similarity / can also do euclid
    dist = cdist(data, data,
                 metric=metric)  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    # create adjacency matrix
    #adj = np.zeros((m, m))

    # cutoff subsetting
    closes = dist < cutoff  # also self links
    # matrix with distance instead boolean: adj[closes] = dist[closes]

    # create df so that names are present
    df_adj = pd.DataFrame(closes, index=df.index.values, columns=df.index.values)
    return df_adj


def to_graph(closes):
    # takes numpy array
    G = nx.from_numpy_matrix(closes.to_numpy())
    # nx.draw(G, edge_color=[i[2]['weight'] for i in G.edges(data=True)])
    # add features?
    return G


def plot_in_out_degree_distributions(edge_index, num_of_nodes, dataset_name):
    """
        Note: It would be easy to do various kinds of powerful network analysis using igraph/networkx, etc.
        I chose to explicitly calculate only the node degree statistics here, but you can go much further if needed and
        calculate the graph diameter, number of triangles and many other concepts from the network analysis field.

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
    fig.subplots_adjust(hspace=0.6)
    '''
    plt.subplot(311)
    plt.plot(in_degrees, color='red')
    plt.xlabel('node id');
    plt.ylabel('in-degree count');
    plt.title('Input degree for different node ids')

    plt.subplot(312)
    plt.plot(out_degrees, color='green')
    plt.xlabel('node id');
    plt.ylabel('out-degree count');
    plt.title('Out degree for different node ids')
    '''
    plt.subplot(313)
    plt.plot(hist, color='blue')
    plt.xlabel('node degree')
    plt.ylabel('# nodes for a given out-degree')
    plt.title(f'Node out-degree distribution for {dataset_name} dataset')
    plt.xticks(np.arange(0, len(hist), 10.0))

    plt.grid(True)
    plt.show()
