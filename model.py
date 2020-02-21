import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

class GCN(nn.Module):
    '''
        “Semi-supervised Classification with Graph Convolutional Networks”
        https://arxiv.org/abs/1609.02907
        
        In paper, two layer
    '''
    def __init__(self, config, in_channels, out_channels):
        '''
            in_channels : num of node features
            out_channels: num of class
        '''
        super().__init__()
        self.config = config
        
        self.hidden_dim = config.hidden_dim
        self.dropout_rate = config.dropout_rate
        
        self.conv1 = gnn.GCNConv(in_channels, self.hidden_dim, improved = False, cached=True, bias=True, normalize=True)
        self.conv2 = gnn.GCNConv(self.hidden_dim, out_channels, improved = False, cached=True, bias=True, normalize=True)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate)
        x = self.conv2(x, edge_index)
        
        return x  