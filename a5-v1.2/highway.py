#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn

class Highway(nn.Module):
    """
    Implementation of Highway Network:
    Using two Linear layer
    """

    def __init__(self, conv_out_length):
        """Init Highway Network
        @param conv_out_length(int): convolution output length
        @param dropout _rate(float): dropout rate
        """
        super(Highway, self).__init__()
        self.conv_out_length = conv_out_length
        self.x_conv_proj_layey = nn.Linear(self.conv_out_length, self.conv_out_length)
        self.gate_layer = nn.Linear(self.conv_out_length, self.conv_out_length)



    def forward(self, conv_out):
        """ Take a mini-batch of convolution output, and compute
        its Highway Network output.

        @param conv_out(Tensor): a tensor of shape (batch_size, conv_out_length)

        @returns x_word_emb(Tensor): a tensor of shape (batch_size, conv_out_length)
        """
        x_proj = torch.relu(self.x_conv_proj_layey(conv_out))
        x_gate = torch.sigmoid(self.gate_layer(conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * conv_out
        return x_highway

### END YOUR CODE

