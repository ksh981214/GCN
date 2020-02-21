import torch
import torch.nn.functional as F

def train(model, data_loader, config):
    learning_rate = config.learning_rate
    l2_reg = config.l2_reg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = config.epochs
    loss_type = config.loss_type
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    
    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            output = model(batch.to(device))
            
            if loss_type == "nll_loss":
                loss = F.nll_loss(F.log_softmax(output,dim=1)[batch.train_mask], batch.y[batch.train_mask])
                
            elif loss_type == "cross_entropy":
                loss = F.cross_entropy(output[batch.train_mask], batch.y[batch.train_mask])
            
            if ((epoch+1)%10)==0:
                print("EPOCH : {} | LOSS : {}".format(epoch+1, loss))
                
            loss.backward()
            optimizer.step()