from torch_geometric.utils import from_networkx
from math import floor

import torch
from torch_geometric.utils import from_networkx
from datetime import date

from utils import *

'''
class TCGA(InMemoryDataset):
    def __init__(self, data_input):
        super(TCGA, self).__init__(data_input)
        self.data, self.slices = torch.load(data_input)
        self.data_input = data_input

    @property
    def raw_file_names(self):
        return ['data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def raw_dir(self):
        return osp.join(r'/media/administrator/INTERNAL3_6TB/TCGA_data', 'LUAD', 'raw')

    @property
    def processed_dir(self):
        return osp.join(r'/media/administrator/INTERNAL3_6TB/TCGA_data', 'LUAD', 'processed')

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        data, slices = self.collate(self.data_input)
        torch.save((data, slices), osp.join(r'/media/administrator/INTERNAL3_6TB/TCGA_data', 'LUAD', 'raw'))

    def __repr__(self):
        return '{}()'.format(self.name)
'''


def create_dataset(datasetname, df_adj=None, df_features=None, df_y=None):
    # returns py torch geometric data object and df with names
    # calculate distance --> get adjacency matrix

    print('Creating Dataset')
    if df_adj is None:
        df_adj = get_adjacency_matrix()

    # convert matrix to G graph object
    graph = to_graph(df_adj)

    # remove self loops
    graph.remove_edges_from(nx.selfloop_edges(graph))

    #draw graph
    #nx.draw_random(graph, arrows= False, with_labels= False, node_size = 100, linewidths= 0.2, width =0.2 )
    #nx.draw_circular(graph, arrows= False, with_labels= False, node_size = 10, linewidths= 0.2, width =0.2 )
    #nx.spring(graph, arrows= False, with_labels= False, node_size = 10, linewidths= 0.2, width =0.2 )

    # convert graph to Pytorch Data object ! missing feautures
    data = from_networkx(graph)
    data.name = datasetname

    if df_features is None:
        df_features = pd.read_csv(r'/media/administrator/INTERNAL3_6TB/TCGA_data/all_binary_selected.txt', index_col=0,
                                  sep='\t')

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

    # could create DATASET object to save format
    # see: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
    
    filepath = f'/media/administrator/INTERNAL3_6TB/TCGA_data/pyt_datasets/{data.name}/raw/numerical_data_{data.num_features}_{date.today()}.pt'

    torch.save(data, filepath)

    return data, filepath


def generate_masks(data, perc_train, perc_test):
    # 70%train, 20% test, remaining 10% val -> Model in Model out
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