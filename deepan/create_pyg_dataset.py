import torch
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import  NormalizeFeatures
from calculate_matrices import *

def create_dataset(df):
    #returns py torch geometric data pbject and df with names
    #calculate distance --> get adjacency matrix
    df_adj = get_adjacency_matrix(0)

    #convert matrix to G graph object
    graph = to_graph(df_adj)

    #convert graph to Pytorch Data object ! missing feautures
    data = from_networkx(graph)

    #create feature matrix
    df = pd.read_csv(r'/media/administrator/INTERNAL3_6TB/TCGA_data/all_binary_selected.txt', index_col=0, sep='\t')

    #sort features fitting to adj matrix
    df = df.reindex(df_adj.index)

    #df to numpy array to Tensor
    x = torch.Tensor(np.array(df))

    #set features in data set
    data.x = x

    #could create DATASET object to save format
    # see: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html

    #save index patient names
    names = pd.Series(df.index.values)
    return data, names
