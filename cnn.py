#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn

class CNN(nn.Module):
    """Implementation of CNN used in character-based encoder
    """

    def __init__(self, char_embedding_size, word_embedding_size, max_word_length, kernel_size = 5):

        super(CNN, self).__init__()
        self.char_embedding_size = char_embedding_size
        self.word_embedding_size = word_embedding_size
        self.kernel_size = kernel_size
        self.max_word_length = max_word_length

        self.conv = nn.Conv1d(self.char_embedding_size,
                              self.word_embedding_size,
                              self.kernel_size)
        self.max_pooling = nn.MaxPool1d(self.max_word_length - self.kernel_size + 1)

    def forward(self, x_reshaped):
        x_conv = self.conv(x_reshaped)
        x_conv_out = self.max_pooling(torch.relu(x_conv)).squeeze(2)
        return x_conv_out

### END YOUR CODE

