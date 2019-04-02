#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()


        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.char_embed_size = 50
        self.max_word_length = 21
        self.dropout_rate = 0.3
        self.vocab = vocab

        self.embeddings = nn.Embedding(num_embeddings = len(vocab.char2id),
                                       embedding_dim = self.char_embed_size,
                                       padding_idx = vocab.char2id['<pad>'])

        self.cnn = CNN(char_embedding_size = self.char_embed_size,
                       word_embedding_size = self.embed_size,
                       max_word_length = self.max_word_length)

        self.highway = Highway(conv_out_length = self.embed_size)
        self.dropout = nn.Dropout(p = self.dropout_rate)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """

        ### YOUR CODE HERE for part 1j
        x_emb = self.embeddings(input) # sentence_length, batch_size, max_word_length, char_embed_size
        sentence_length, batch_size, max_word_length, char_embed_size = x_emb.shape
        view_shape = (sentence_length * batch_size, max_word_length, char_embed_size)
        x_reshaped = x_emb.view(view_shape).transpose(1, 2) # sentence_length * batch_size, char_embed_size, max_word_length
        x_conv_out = self.cnn(x_reshaped) # sentence_length * batch_size, embed_size
        x_word_emb = self.dropout(self.highway(x_conv_out)) # sentence_length * batch_size, embed_size
        output = x_word_emb.view(sentence_length, batch_size, -1)
        return output

        ### END YOUR CODE

