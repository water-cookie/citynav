import gzip
import json

import torch
import torch.nn as nn
from torch import Tensor

from vlnce.defaultpaths import WORD_EMBEDDING_PATH


class InstructionEncoder(nn.Module):
    def __init__(
        self,
        embedding_size: int = 50,
        hidden_size: int = 128,
        rnn_type: str = 'LSTM',
        final_state_only: bool = True,
        bidirectional: bool = False,
        use_pretrained_embeddings: bool = False,
        fine_tune_embeddings: bool = False,
        vocab_size: int = 2054,
        embedding_file: str = str(WORD_EMBEDDING_PATH),
    ):
        """An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            embedding_size: The dimension of each embedding vector
            hidden_size: The hidden (output) size
            rnn_type: The RNN cell type.  Must be GRU or LSTM
            final_state_only: If True, return just the final state
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.final_state_only = final_state_only
        self.bidirectional = bidirectional

        assert rnn_type in ['GRU', 'LSTM']
        rnn = nn.GRU if self.rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn(
            input_size=embedding_size,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
        )

        if use_pretrained_embeddings:
            self.embedding_layer = nn.Embedding.from_pretrained(
                embeddings=self._load_embeddings(embedding_file),
                freeze=not fine_tune_embeddings,
            )
        else:  # each embedding initialized to sampled Gaussian
            self.embedding_layer = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_size,
                padding_idx=0,
            )

    @property
    def output_size(self):
        return self.hidden_size * (1 + int(self.bidirectional))

    def _load_embeddings(self, embedding_file) -> Tensor:
        """Loads word embeddings from a pretrained embeddings file.
        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged: https://bit.ly/3u3hkYg
        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, observations) -> Tensor:
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        instruction = observations["instruction"].long()
        lengths = (instruction != 0.0).long().sum(dim=1)
        instruction = self.embedding_layer(instruction)


        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1).cpu()

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            instruction, lengths, batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)

        if self.rnn_type == "LSTM":
            final_state = final_state[0]

        if self.final_state_only:
            return final_state.squeeze(0)
        else:
            return nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )[0].permute(0, 2, 1)
