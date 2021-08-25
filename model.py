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

class image_embedding(nn.Module):
    def __init__(self, in_channels: int = 3, img_size: int = 224, patch_size: int = 16, emb_dim: int = 16*16*3):
        super().__init__()

        self.rearrange = Rearrange('b c (num_w p1) (num_h p2) -> b (num_w num_h) (p1 p2 c) ', p1=patch_size, p2=patch_size)
        self.linear = nn.Linear(in_channels * patch_size * patch_size, emb_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        n_patches = img_size * img_size // patch_size**2
        self.positions = nn.Parameter(torch.randn(n_patches + 1, emb_dim))

    def forward(self, x):
        batch, channel, width, height = x.shape

        x = self.rearrange(x) # flatten patches 
        x = self.linear(x) # embedded patches 

        # ================ ToDo1 ================ #
        # (1) Build [token; image embedding] by concatenating class token with image embedding
        c = repeat(self.cls_token, '() n d -> b n d', b=batch) 
        x = torch.cat([c, x], dim=1)

        # (2) Add positional embedding to [token; image embedding]
        x += self.positions    
        # ======================================= #
        return x

class multi_head_attention(nn.Module):
    def __init__(self, emb_dim: int = 16*16*3, num_heads: int = 8, dropout_ratio: float = 0.2, verbose = False, **kwargs):
        super().__init__()
        self.v = verbose

        self.emb_dim = emb_dim 
        self.num_heads = num_heads 
        self.scaling = (self.emb_dim // num_heads) ** -0.5
        
        self.value = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)
        self.att_drop = nn.Dropout(dropout_ratio)

        self.linear = nn.Linear(emb_dim, emb_dim)
                
    def forward(self, x: Tensor) -> Tensor:
        # query, key, value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        if self.v: print(Q.size(), K.size(), V.size()) 

        # q = k = v = patch_size**2 + 1 & h * d = emb_dim
        Q = rearrange(Q, 'b q (h d) -> b h q d', h=self.num_heads)
        K = rearrange(K, 'b k (h d) -> b h d k', h=self.num_heads)
        V = rearrange(V, 'b v (h d) -> b h v d', h=self.num_heads)
        if self.v: print(Q.size(), K.size(), V.size()) 

        ## scaled dot-product
        weight = torch.matmul(Q, K) 
        weight = weight * self.scaling
        if self.v: print(weight.size()) 
        
        attention = torch.softmax(weight, dim=-1)
        attention = self.att_drop(attention) 
        if self.v: print(attention.size())

        context = torch.matmul(attention, V) 
        context = rearrange(context, 'b h q d -> b q (h d)')
        if self.v: print(context.size())

        x = self.linear(context)
        return x , attention

class mlp_block(nn.Module):
    def __init__(self, emb_dim: int = 16*16*3, forward_dim: int = 4, dropout_ratio: float = 0.2, **kwargs):
        super().__init__()
        self.linear_1 = nn.Linear(emb_dim, forward_dim * emb_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear_2 = nn.Linear(forward_dim * emb_dim, emb_dim)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x) 
        x = self.linear_2(x)
        return x

class encoder_block(nn.Sequential):
    def __init__(self, emb_dim:int = 16*16*3, num_heads:int = 8, forward_dim: int = 4, dropout_ratio:float = 0.2):
        super().__init__()

        self.norm_1 = nn.LayerNorm(emb_dim)
        self.mha = multi_head_attention(emb_dim, num_heads, dropout_ratio)

        self.norm_2 = nn.LayerNorm(emb_dim)
        self.mlp = mlp_block(emb_dim, forward_dim, dropout_ratio)

    def forward(self, x):
        # ================ ToDo2 ================ #
        # x = normalize (input)
        _x = self.norm_1(x)
        # x', attention = multihead_attention (x)
        _x, attention = self.mha(_x)
        # x = x' + residual(x)
        x = _x + x
        
        # x' = normalize(x)
        _x = self.norm_2(x)
        # x' = mlp(x')
        _x = self.mlp(_x)
        # x  = x' + residual(x)
        x = _x + x
        # ======================================= #
        return x, attention

class vision_transformer(nn.Module):
    """ Vision Transformer model
    classifying input images (x) into classes
    """
    def __init__(self, in_channel: int = 3, img_size:int = 224, 
                 patch_size: int = 16, emb_dim:int = 16*16*3, 
                 n_enc_layers:int = 15, num_heads:int = 3, 
                 forward_dim:int = 4, dropout_ratio: float = 0.2, 
                 n_classes:int = 1000):
        super().__init__()


        # ================ ToDo3 ================ #
        # You need (1) image embedding module & (2) encoder module
        self.ie = image_embedding(in_channel, img_size, patch_size, emb_dim)
        self.em = nn.ModuleList([
            encoder_block(emb_dim, num_heads, forward_dim, dropout_ratio) for _ in range(n_enc_layers)
        ])
        # ======================================= #      

        self.normalization = nn.LayerNorm(emb_dim)
        self.classification_head = nn.Linear(emb_dim, n_classes) 

    def forward(self, x):
        # ================ ToDo3 ================ #
        # (1) image embedding
        x = self.ie(x)

        # (2) transformer_encoder
        attentions = []
        for encoder in self.em:
            x, attention = encoder(x)
            attentions.append(attention)
        # ======================================= #

        x = x[:, 0, :] # cls_token output
        x = self.normalization(x)
        x = self.classification_head(x)

        return x, attentions

# model = vision_transformer(
#     in_channel = 3, 
#     img_size = 384,
#     patch_size = 4,
#     emb_dim = 3*5,
#     num_heads = 3,
#     n_enc_layers = 2,
#     forward_dim= 4,
#     dropout_ratio = 0.2,
#     n_classes = 18)
# print(model)
#summary(model, input_size=(3, 384, 384), device='cpu')
