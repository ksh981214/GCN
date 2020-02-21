class config():    
    #Preprocess
    root_dir = './data/'
    batch_size = 1
    
    #Model
    hidden_dim = 16 
    dropout_rate = 0.5
    l2_reg = 5*10e-4
    
    #Train
    learning_rate = 0.01
    epochs = 200
    loss_type = 'cross_entropy' # or nll_loss