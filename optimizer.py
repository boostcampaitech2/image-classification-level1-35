import torch

def get_optimizer(model, config):
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr)
    elif config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)

    return optimizer

def get_scheduler(optimizer, config):
    if config.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 10,
            eta_min = 0  
        )
    elif config.scheduler == 'LRscheduler':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                        lr_lambda=lambda epoch: 0.95 ** epoch,
                        last_epoch=-1,
                        verbose=False)

    return scheduler