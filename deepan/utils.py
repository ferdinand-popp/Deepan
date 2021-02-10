import sys
import numpy

def print_edge_pairs(edge_index):
    numpy.set_printoptions(threshold=sys.maxsize)
    print(edge_index.numpy().T)