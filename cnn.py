#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, c_embed, w_embed,k=5):
        super(CNN, self).__init__()
        self.char_embed = c_embed
        self.word_embed = w_embed
        self.k = k
        self.conv = nn.Conv1d(in_channels=c_embed, out_channels=w_embed, kernel_size=k, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv(x)
        conv_out, _ = torch.max(self.relu(conv), 2)
        return conv_out


### END YOUR CODE

