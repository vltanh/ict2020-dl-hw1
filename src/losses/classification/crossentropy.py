import torch
import torch.nn as nn

__all__ = ['BCELoss', 'BCEWithLogitsLoss', 'CrossEntropyLoss']


class BCELoss(nn.BCELoss):
    def __call__(self, output, target):
        target = target.type_as(output)
        if len(target.shape) != len(output.shape):
            target = target.unsqueeze(1)
        return super().__call__(output, target)


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __call__(self, output, target):
        target = target.type_as(output)
        if len(target.shape) != len(output.shape):
            target = target.unsqueeze(1)
        return super().__call__(output, target)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, **kwargs):
        if weight is not None:
            weight = torch.FloatTensor(weight)
        super().__init__(weight, **kwargs)
