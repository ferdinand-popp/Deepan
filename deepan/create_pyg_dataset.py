from math import floor

from torch_geometric.utils import from_networkx
from datetime import date

from utils import *


def create_dataset(datasetname, df_adj=None, df_features=None, df_y=None):
    """
    Creates a pytorch data object from the inputs features, adjacency and survival data.
    Returns: data object, filepath (where it was saved)
    """

    print('Creating Dataset')

    # convert matrix to G graph object
    graph = to_graph(df_adj)

    # remove self loops (identity)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # possibly draw graph draw
    draw_graph_inspect(graph)

    # convert graph to Pytorch Data object ! missing feautures
    data = from_networkx(graph)
    data.name = datasetname

    # sort features fitting to adj matrix
    df_features = df_features.reindex(df_adj.index)

    # df to numpy array to Tensor
    x = torch.Tensor(np.array(df_features))

    # set features in data set
    data.x = x

    # set y labels added later in supervised way
    data.y = None
    data.weight = None
    data.adj_self = df_adj

    if df_y is not None:
        data.survival = df_y.reindex(df_adj.index)

    filepath = f'/media/administrator/INTERNAL3_6TB/TCGA_data/pyt_datasets/{data.name}/raw/numerical_data_{data.num_features}_{date.today()}.pt'
    torch.save(data, filepath)

    return data, filepath


def generate_masks(data, perc_train, perc_test):
    # Node masks: 70%train, 20% test, remaining 10% val -> Model in Model out
    print('Generating masks for Nodes')

    nodes = data.num_nodes
    num_train = floor(nodes * perc_train)  # round down to whole number
    num_test = floor(nodes * perc_test)
    list_train = [True if x in list(range(0, num_train)) else False for x in list(range(0, nodes))]
    list_test = [True if x in list(range(num_train, num_test)) else False for x in list(range(0, nodes))]
    list_val = [True if x in list(range(num_train + num_test, nodes)) else False for x in list(range(0, nodes))]
    data.train_mask = list_train
    data.test_mask = list_test
    data.val_mask = list_val
    return data
