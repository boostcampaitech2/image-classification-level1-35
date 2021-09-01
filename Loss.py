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

class LabelSmoothingLoss(nn.Module):
    def __init__(self, config, classes=3, smoothing=0.05, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = config.num_classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class CrossEntropy_FoscalLoss_LabelSmoothingLoss(nn.Module):
    def __init__(self, class_weights, config):
        super().__init__()
        self.weight1 = config.loss1_weight
        self.weight2 = config.loss2_weight
        self.weight3 = config.loss3_weight
        self.device = config.device
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights).to(self.device, dtype=torch.float))(inputs, targets)
        fs_loss = FocalLoss()(inputs, targets)
        ls_loss = LabelSmoothingLoss()(inputs, targets)
        return ce_loss * self.weight1 + fs_loss * self.weight2 + ls_loss * self.weight3

def get_loss(config, class_weigth):
    if config.loss == 'CrossEntropy':
        loss_func1 = torch.nn.CrossEntropyLoss()
    elif config.loss == 'Crossentropy_foscal':
        loss_func1 = CrossEntropy_FoscalLoss(class_weigth, config)
    elif config.loss == 'CrossEntropy_weighted':
        loss_func1 = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weigth).to(config.device, dtype=torch.float))
    elif config.loss == 'Foscal':
        loss_func1 = FocalLoss()
        
    return loss_func1