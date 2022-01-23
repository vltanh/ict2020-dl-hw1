from typing import List

import torch
import torch.nn as nn

__all__ = ['MLPClassifier']


class MLPClassifier(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims: List,
                 output_dim,
                 activation=None,
                 dropout=None):
        super().__init__()

        hidden_dims = [input_dim, *hidden_dims]
        layers = [
            nn.Sequential(
                nn.Linear(di, do),
                nn.ReLU() if activation == 'relu' else nn.Identity(),
            ) if dropout is None else nn.Sequential(
                nn.Linear(di, do),
                nn.Dropout(dropout),
                nn.ReLU() if activation == 'relu' else nn.Identity(),
            )
            for di, do in zip(hidden_dims, hidden_dims[1:])
        ]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))
