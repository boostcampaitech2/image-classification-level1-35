import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def get_loss(config, class_weigth):
    if config.loss == 'CrossEntropy':
        loss_func1 = torch.nn.CrossEntropyLoss()
    elif config.loss == 'Crossentropy_foscal':
        loss_func1 = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weigth).to(config.device, dtype=torch.float))
        loss_func2 = FocalLoss()
        return loss_func1, loss_func2
    elif config.loss == 'CrossEntropy_weighted':
        loss_func1 = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weigth).to(config.device, dtype=torch.float))
    elif config.loss == 'Foscal':
        loss_func1 = FocalLoss()
        
    return loss_func1