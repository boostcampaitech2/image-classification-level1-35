import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models  as cvmodels
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from torchsummary import summary
import timm

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(MyModel, self).__init__()
        self.pretrained = cvmodels.resnet50(pretrained=True)
        #pretrained = cvmodels.resnet50(pretrained=True)
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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


#model = MyModel(num_classes=18)
#print(model)
#summary(model, input_size=(3, 512, 384), device='cpu')
# swsl_resnext50_32x4d
class Efficientnet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.pretrained = timm.create_model('efficientnet_b3a', pretrained=True)
        self.pretrained.classifier = nn.Sequential(
            nn.Linear(1536, 512, bias=True),
            nn.LeakyReLU(0.3),
            nn.Linear(512, num_classes, bias=True),
            # nn.LeakyReLU(0.3),
            # nn.Linear(256, 18, bias=True),
        )
        #nn.init.xavier_uniform(self.efficient.classifier.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained(x)
        return x

#model = Efficientnet(num_classes=18)
#print(model)
#summary(model, input_size=(3, 256, 256), device='cpu')

class SWSLResnext50(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.pretrained = timm.create_model('swsl_resnext50_32x4d', pretrained=True)
        self.pretrained.fc = nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.LeakyReLU(0.3),
            nn.Linear(512, 18, bias=True),
            # nn.LeakyReLU(0.3),
            # nn.Linear(256, 18, bias=True),
        )
        #nn.init.xavier_uniform(self.efficient.classifier.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained(x)
        return x

# model = SWSLResnext50(num_classes=18)
# print(model)
#summary(model, input_size=(3, 256, 256), device='cpu')