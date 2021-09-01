import torch
import timm


model =  timm.create_model('efficientnet_b3a', pretrained=True)

# print(model.state_dict().keys())
print(model)