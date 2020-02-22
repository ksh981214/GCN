from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt

def train(model, data_loader, config):
    def check_early_stop(losses, threshold):
        if len(losses) >= 10:
            if np.mean(losses[:-10]) > threshold:
                return True, np.mean(losses[:-10])
            else:
                return False, np.mean(losses[:-10])
            
    def show_loss_plt(x_plot, train_losses, val_losses):
        now = datetime.now()
        plt.plot(x_plot, train_losses)
        plt.plot(x_plot, val_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Training','Validation'])
        plt.savefig('./result/loss_{}.png'.format(str(now)))
        plt.show()
        
    def show_acc_plt(x_plot, train_acc_plot, val_acc_plot):
        now = datetime.now()
        plt.plot(x_plot, train_acc_plot)
        plt.plot(x_plot, val_acc_plot)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Training','Validation'])
        plt.savefig('./result/acc_{}.png'.format(str(now)))
        plt.show()
        
            
    learning_rate = config.learning_rate
    l2_reg = config.l2_reg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = config.epochs
    loss_type = config.loss_type
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    
    model.train()
    
    x_plot = []
    train_loss_plot = []
    val_loss_plot = []
    
    train_acc_plot = []
    val_acc_plot = []
    
    early_stop=False
    val_loss_mean = 10000
    for epoch in range(epochs):
        if early_stop:
            break
        for batch in data_loader:
            optimizer.zero_grad()
            output = model(batch.to(device))
            
            # Tranining Loss
            if loss_type == "nll_loss":
                train_loss = F.nll_loss(F.log_softmax(output,dim=1)[batch.train_mask], batch.y[batch.train_mask])
                
            elif loss_type == "cross_entropy":
                train_loss = F.cross_entropy(output[batch.train_mask], batch.y[batch.train_mask])

            train_loss.backward()
            optimizer.step()
            
            # Validation Loss
            if loss_type == "nll_loss":
                val_loss = F.nll_loss(F.log_softmax(output,dim=1)[batch.val_mask], batch.y[batch.val_mask])
                
            elif loss_type == "cross_entropy":
                val_loss = F.cross_entropy(output[batch.val_mask], batch.y[batch.val_mask])
                
            if ((epoch+1)%10)==0:
                print("EPOCH : {:<4} | TRAIN LOSS : {:<.5} | VAL LOSS : {:<.5}".format(epoch+1, train_loss, val_loss))
                
            # Validation Accuracy
            _, pred = output.max(dim=1)
            
            train_correct = float(pred[batch.train_mask].eq(batch.y[batch.train_mask]).sum().item())
            train_acc = train_correct / batch.train_mask.sum().item()
            
            val_correct = float(pred[batch.val_mask].eq(batch.y[batch.val_mask]).sum().item())
            val_acc = val_correct / batch.val_mask.sum().item()
                
            #Plot
            x_plot.append(epoch)
            train_loss_plot.append(train_loss.item())
            val_loss_plot.append(val_loss.item())
            
            train_acc_plot.append(train_acc)
            val_acc_plot.append(val_acc)
            
            # Early Stop
            if len(val_loss_plot) >=10:
                early_stop, val_loss_mean = check_early_stop(val_loss_plot, val_loss_mean)
    
    show_loss_plt(x_plot, train_loss_plot, val_loss_plot)
    show_acc_plt(x_plot, train_acc_plot, val_acc_plot)