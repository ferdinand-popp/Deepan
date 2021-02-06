import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import networkx as nx

def get_adjacency_matrix(df):
    # takes binary matrix,calculates distance and cutoffs -> returns boolean distance df
    df = pd.read_csv(r'/media/administrator/INTERNAL3_6TB/TCGA_data/all_binary_selected.txt', index_col=0, sep='\t')

    data = df.to_numpy()
    n, m = data.shape

    # generate distance matrix on cosine similarity / can also do euclid
    dist = cdist(data, data,
                 metric='cosine')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    # create adjacency matrix
    adj = np.zeros((m, m))

    # cutoff subsetting
    closes = dist < 0.3
    # matrix with distnace instead boolean: adj[closes] = dist[closes]

    #create df so that names are present
    df_adj = pd.DataFrame(closes, index= df.index.values, columns= df.index.values)
    return df_adj #numpy array


def to_graph(closes):
    #takes numpy array
    G = nx.from_numpy_matrix(closes.to_numpy())
    #nx.draw(G, edge_color=[i[2]['weight'] for i in G.edges(data=True)])
    #add features?
    return G