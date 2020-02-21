from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

class Preprocess():
    def __init__(self, config, d_name):
        self.root_dir = config.root_dir
        self.batch_size = config.batch_size
        
        #self.cora = Planetoid(root='./data/cora', name='Cora')
        #self.citeseer = Planetoid(root='./data/citeseer', name='CiteSeer')
        #self.pubmed = Planetoid(root='./data/pubmed', name='PubMed')
        
        self.num_classes, self.num_node_features, self.data = self.get_data(d_name)
        
    def get_data(self, d_name):
        '''
            d_name = 'Cora', 'CiteSeer', 'PubMed'
        '''
        dataset = Planetoid(root=self.root_dir + d_name , name=d_name)
        
        return dataset.num_classes, dataset.num_node_features, DataLoader(dataset, batch_size = self.batch_size)