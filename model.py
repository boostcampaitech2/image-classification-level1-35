import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models  as cvmodels
import timm
import math

class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s = 10, m = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        cos_th = cos_th.type(torch.float32)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m   
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs


class SHOPEEDenseNet(nn.Module):

    def __init__(self, channel_size, out_feature, dropout=0.5, backbone='efficientnet_b3a', pretrained=True):
        super(SHOPEEDenseNet, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.backbone.global_pool = nn.Identity()
        self.backbone.classifier = nn.Identity()
        self.channel_size = channel_size
        self.out_feature = out_feature
        self.in_features = 1536
        self.margin = ArcModule(in_features=self.channel_size, out_features = self.out_feature)
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.fc1 = nn.Linear(self.in_features * 7 * 7 , self.channel_size)
        self.bn2 = nn.BatchNorm1d(self.channel_size)
        
    def forward(self, x, labels=None):
        features = self.backbone(x)
        features = self.bn1(features)
        features = self.dropout(features)
        #print(features.shape)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.bn2(features)
        features = F.normalize(features)
        if labels is not None:
            return self.margin(features, labels)
        return features

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(MyModel, self).__init__()
        self.pretrained = cvmodels.resnet50(pretrained=True)
        self.pretrained.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 1000),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Efficientnet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.pretrained = timm.create_model('efficientnet_b3a', pretrained=True)
        self.pretrained.classifier = nn.Sequential(
            nn.Linear(1536, 512, bias=True),
            nn.LeakyReLU(0.3),
            nn.Linear(512, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained(x)
        return x

class SWSLResnext50(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.pretrained = timm.create_model('swsl_resnext50_32x4d', pretrained=True)
        self.pretrained.fc = nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.LeakyReLU(0.3),
            nn.Linear(512, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained(x)
        return x

class MobilenetV2(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.pretrained = timm.create_model('mobilenetv2_100', pretrained=True)
        self.pretrained.classifier = nn.Sequential(
            nn.Linear(1280, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained(x)
        return x

class VIT(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.pretrained = timm.create_model('vit_base_patch16_384', pretrained=True)
        self.pretrained.head = nn.Sequential(
            nn.Linear(768, 256, bias=True),
            nn.LeakyReLU(0.3),
            nn.Linear(256, num_classes, bias=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained(x)
        return x

class VGG19(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.pretrained = timm.create_model('vgg19_bn', pretrained=True)
        self.pretrained.head.fc = nn.Sequential(
            nn.Linear(4096, 512, bias=True),
            nn.LeakyReLU(0.3),
            nn.Linear(512, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained(x)
        return x

class Efficientnet_b0(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.pretrained = timm.create_model('efficientnet_b0', pretrained=True)
        self.pretrained.classifier = nn.Sequential(
            nn.Linear(1280, 512, bias=True),
            nn.LeakyReLU(0.3),
            nn.Linear(512, num_classes, bias=True),
        )

def get_model(config):
    if config.model_name == 'efficientnet_b3a':
        model = Efficientnet(num_classes=config.num_classes).to(config.device)
    elif config.model_name == 'Efficientnet_b0':
        model = Efficientnet_b0(num_classes=config.num_classes).to(config.device)
    elif config.model_name == 'swsl_resnext50_32x4d':
        model = SWSLResnext50(num_classes=config.num_classes).to(config.device)
    elif config.model_name == 'mobilenetv2_100':
        model = MobilenetV2(num_classes=config.num_classes).to(config.device)     
    elif config.model_name == 'vgg19_bn':
        model = VGG19(num_classes=config.num_classes).to(config.device)     
    elif config.model_name == 'vit_base_patch16_384':
        model = VIT(num_classes=config.num_classes).to(config.device)     
    elif config.model_name == 'SHOPEEDenseNet':
        model = SHOPEEDenseNet(224, out_feature=config.num_classes).to(config.device)    
    return model