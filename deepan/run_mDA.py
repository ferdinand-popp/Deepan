import numpy as np
import torch
from mDA_implementation import mDA
filepath = r'/media/administrator/INTERNAL3_6TB/TCGA_data/pyt_datasets/NSCLC/raw/numerical_data_308_2021-03-18.pt'

data = torch.load(filepath)
print('Try mDA')
#data.x.numpy.tofile('data_xx.csv', sep = ',')
#data.adj_self.numpy.tofile('data_A_n.csv', sep = ',')

xx = data.x.cpu().detach().numpy().T
A_n = data.adj_self.to_numpy()

ret = mDA(xx, 0.2, 0.005, A_n)
print(ret)
