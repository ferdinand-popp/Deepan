import torch
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import  NormalizeFeatures
from calculate_matrices import *
from math import floor

def create_dataset(df):
    #returns py torch geometric data pbject and df with names
    #calculate distance --> get adjacency matrix
    df_adj = get_adjacency_matrix(0)

    #convert matrix to G graph object
    graph = to_graph(df_adj)

    #remove self loops
    graph.remove_edges_from(nx.selfloop_edges(graph))

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

    #set y labels added later in supervised way
    data.y = []
    data.weight = None

    #could create DATASET object to save format
    # see: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html

    #save index patient names
    names = pd.Series(df.index.values)
    return data, names

def generate_masks(data, perc_train, perc_test):
    #70%train, 20% test, remaining 10% val
    nodes = data.num_nodes
    num_train = floor(nodes*perc_train) #round down to whole number
    num_test = floor(nodes*perc_test)
    list_train = [True if x in list(range(0,num_train)) else False for x in list(range(0,nodes))]
    list_test = [True if x in list(range(num_train, num_test)) else False for x in list(range(0,nodes))]
    list_val = [True if x in list(range(num_train + num_test, nodes)) else False for x in list(range(0,nodes))]
    data.train_mask = list_train
    data.test_mask = list_test
    data.val_mask = list_val
    return data