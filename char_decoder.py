#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(input_size = char_embedding_size,
                                   hidden_size = hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(num_embeddings = len(target_vocab.char2id),
                                           embedding_dim = char_embedding_size,
                                           padding_idx = target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab

        ### END YOUR CODE



    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters.
                           A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters.
                             A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        input = self.decoderCharEmb(input)
        output, dec_hidden = self.charDecoder(input, dec_hidden) # output : seq_len, batch_size, char_embed_dim
        s_t = self.char_output_projection(output) # s_t : seq_len, batch_size, V_char
        return s_t, dec_hidden

        ### END YOUR CODE


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch).
                              Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder.
                           A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        # TODO: Check loss implementation
        s_t, dec_hidden = self.forward(char_sequence[:-1], dec_hidden) # s_t: length-1, batch, self.vocab_size
        #target = char_sequence[1:] # length-1, batch
        target = char_sequence[1:].contiguous().view(-1)
        s_t  = s_t.view(-1, s_t.shape[-1])
        loss = nn.CrossEntropyLoss()
        output = loss(s_t, target)

        return output

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].shape[1]
        dec_hidden = initialStates

        input = torch.LongTensor([self.target_vocab.start_of_word for _ in range(batch_size)], device=device).view(1, -1)
        end = self.target_vocab.end_of_word
        dec_char = [["", False] for _ in range(batch_size)]

        for i in range(max_length):
            output, dec_hidden = self.forward(input, dec_hidden)
            input = output.argmax(2)

            for batch_index, temp_char_index in enumerate(input.squeeze(0).view(-1).data.tolist()):
                if not dec_char[batch_index][1]:
                    if temp_char_index != end:
                        dec_char[batch_index][0] += self.target_vocab.id2char[temp_char_index]
                    else:
                        dec_char[batch_index][1] = True

        output_word = [i[0] for i in dec_char]
        return output_word

        ### END YOUR CODE


