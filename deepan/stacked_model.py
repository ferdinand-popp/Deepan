import torch
from torch import nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt

'''
From https://github.com/ShayanPersonal/stacked-autoencoder-pytorch and PyT-geometric
'''


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class CDAutoEncoder(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.
    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
    """

    def __init__(self, input_size, output_size):
        super(CDAutoEncoder, self).__init__()

        self.forward_pass = LinearEncoder(in_channels=input_size, out_channels=output_size)
        # self.backward_pass = nn.Sequential(
        #     nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=2, padding=0),
        #     nn.ReLU(),
        # )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x, edge_index):
        # Train each autoencoder individually
        x = x.clone().detach()
        # Add noise, but use the original lossless input as the target.
        # x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        y = self.forward_pass(x, edge_index)

        if self.training:
            # x_reconstruct = self.backward_pass(y)
            loss = self.criterion(y, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return y.clone().detach()

    # def reconstruct(self, x):
    #    return self.backward_pass(x)


class StackedAutoEncoder(nn.Module):
    r"""
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self, input_size, output_size):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = CDAutoEncoder(input_size=input_size, output_size=output_size)
        self.ae2 = CDAutoEncoder(input_size=input_size, output_size=output_size)
        self.ae3 = CDAutoEncoder(input_size=input_size, output_size=output_size)

    def forward(self, x, edge_index):
        a1 = self.ae1(x, edge_index)
        a2 = self.ae2(a1, edge_index)
        a3 = self.ae3(a2, edge_index)

        if self.training:
            return a3

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.forward(*args, **kwargs)


def corrupt(clean_data, noise):
    """
        Input noise for the MGAE
        May need to be tensor and not array.
        """
    data = np.copy(clean_data)
    data[np.random.sample(size=data.shape) < noise] = 0
    return data

#Data import
#data = torch.load(r'/media/administrator/INTERNAL3_6TB/TCGA_data/pyt_datasets/NSCLC/raw/numerical_data_308_2021-03-18.pt')
data = torch.load(r'/home/fpopp/PycharmProjects/Deepan/data/Planetoid/Cora/processed/data.pt')[0]
num_features = data.num_features
features = data.x
edge_index = data.edge_index

#Model setup
model = StackedAutoEncoder(num_features, num_features)
epochs = 100
criterion = nn.MSELoss()
# not necessary as no overall optimizing is done:
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

losses = []
for epoch in range(epochs):
    model.train()
    # corrupt input feature matrix
    x = corrupt(features, 0.1)
    x = torch.from_numpy(x)
    # forward info get hidden embedding
    z = model.encode(x, edge_index)
    # difference complete
    loss = criterion(z, features)
    losses.append(float(loss))
    print(f'Epoch:{epoch} Loss:{loss}')

    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()

plt.plot(losses)
plt.show()
plt.savefig('Testrun.png')