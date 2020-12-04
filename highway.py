#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn

class Highway(nn.Module):

    def __init__(self, embed_word, dropout_rate=0.3):
        super(Highway, self).__init__()
        self.projection = nn.Linear(in_features=embed_word, out_features=embed_word, bias=True)
        self.relu = nn.ReLU()
        self.gate = nn.Linear(in_features=embed_word, out_features=embed_word, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.relu(self.projection(x))
        gate = self.sigmoid(self.gate(x))
        highway=proj*gate+(1-gate)*x
        return self.dropout(highway)


### END YOUR CODE 

