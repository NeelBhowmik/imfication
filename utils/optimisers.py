import torch.optim as optim

def optimisers(optim_name, model, lr=0.001, momentum=0.5, weight_decay=0.0001):
    if optim_name == 'SGD':
        optimizer = optim.__dict__[optim_name](
            model.parameters(), 
            lr=lr, momentum=momentum)
    elif optim_name == 'Adam':
        optimizer = optim.__dict__[optim_name](
            model.parameters(), 
            lr=lr, weight_decay=weight_decay)
    else:
        print(f'{optim_name} Optimiser not found: available [SGD, Adam]')
        exit()
    
    return optimizer