import torch
import timm


model =  timm.create_model('vit_base_patch16_384', pretrained=True)

print(model.state_dict().keys())