import torch

def test(model, data_loader, config):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    for batch in data_loader:
        _, pred = model(batch.to(device)).max(dim=1)
        
        correct = float (pred[batch.test_mask].eq(batch.y[batch.test_mask]).sum().item())
        
        acc = correct / batch.test_mask.sum().item()
        
        print('Accuracy: {:.4f}'.format(acc))