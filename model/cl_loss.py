import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss


class CLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = CrossEntropyLoss()

    def forward(self, x: Tensor):
        label = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        loss = self.loss(x, label)
        return loss
