import func
import dataset

model = timm.create_model('tf_efficientnet_b0', num_classes=18, pretrained=True)
model.classifier = nn.Linear(in_features=1280, out_features=18, bias=True)


# for n, p in model.named_parameters():
    # if '' not in n:
        # p.requires_grad = False


# class MyModel(nn.Module):
#     def __init__(self, model, num_classes: int = 18, dr_rate=0.5):
#         super().__init__()
#         self.pretrained = model
#         self.dr_rate = dr_rate
#         self.classifier.classifier = nn.Linear(1280, num_classes, bias=True)
#         if dr_rate:
#             self.dropout = nn.Dropout(p=dr_rate)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pretrained(x)
#         x = torch.flatten(x, 1)
#         if self.dr_rate:
#             x = self.dropout(x)
#         x = self.classifier(x)
#         return x


#loss함수
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