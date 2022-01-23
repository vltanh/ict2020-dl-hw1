import torch.nn as nn


class MSELoss(nn.MSELoss):
    def __call__(self, output, target):
        target = target.type_as(output)
        if len(target.shape) != len(output.shape):
            target = target.unsqueeze(1)
        return super().__call__(output, target)
