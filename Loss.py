import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
import torch.functional as F

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

class CrossEntropy_FoscalLoss(nn.Module):
    def __init__(self, class_weights, config):
        super().__init__()
        self.weight1 = config.loss1_weight
        self.weight2 = config.loss2_weight
        self.device = config.device
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights).to(self.device, dtype=torch.float))(inputs, targets)
        fs_loss = FocalLoss()(inputs, targets)
        return ce_loss * self.weight1 + fs_loss * self.weight2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

def get_loss(config, class_weigth):
    if config.loss == 'CrossEntropy':
        loss_func1 = torch.nn.CrossEntropyLoss()
    elif config.loss == 'Crossentropy_foscal':
        loss_func1 = CrossEntropy_FoscalLoss(class_weigth, config)
    elif config.loss == 'CrossEntropy_weighted':
        loss_func1 = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weigth).to(config.device, dtype=torch.float))
    elif config.loss == 'Foscal':
        loss_func1 = FocalLoss()
    elif config.loss == 'MSE':
        loss_func1 = MSELoss()
        
    return loss_func1