from config import config
from preprocess import Preprocess
from model import GCN
from train import train
from test import test

config=config()

def main():
    p = Preprocess(config, 'Cora')
    
    model = GCN(config, p.num_node_features, p.num_classes)
    train(model, p.data, config)
    test(model, p.data, config)
    
main()