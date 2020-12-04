#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

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

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f

        c_embed=50
        pad_token_id=vocab.char2id['<pad>']
        self.embed_size=embed_size
        self.embeddings=nn.Embedding(len(vocab.char2id), c_embed, padding_idx=pad_token_id)
        self.cnn = CNN(c_embed=c_embed, w_embed=embed_size, k=5)
        self.highway = Highway(embed_word=embed_size, dropout_rate=0.3)

        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f

        [sentence_length, batch_size, max_word_length] = list(input_tensor.size())
        input_batch = input_tensor.contiguous().view(-1, max_word_length)

        x_emb = self.embeddings(input_batch)
        x_emb = x_emb.permute(0, 2, 1)

        x_conv_out = self.cnn.forward(x_emb)

        x_word_emb = self.highway(x_conv_out)

        x_word_emb_unbatched = x_word_emb.view(sentence_length, batch_size, -1)

        return x_word_emb_unbatched



        ### END YOUR CODE
