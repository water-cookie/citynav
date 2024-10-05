import gzip
import json
import torch
import torch.nn as nn
from torch import Tensor

from vlnce.defaultpaths import WORD_EMBEDDING_PATH
from vlnce.models.encoders.rnn_state_encoder import build_rnn_state_encoder

from .goal_predictor import MapEncoder
from .ddppo.resenet_encoders import TorchVisionResNet50, ResnetDepthEncoder

class Seq2SeqwithMap(nn.Module):

    def __init__(
        self,
        map_size: int,
        state_encoder_hidden_size=512,
        state_encoder_rnn_type='GRU',
    ):
        super().__init__()
        self.state_encoder_hidden_size = state_encoder_hidden_size

        self.instruction_encoder = InstructionEncoder(vocab_size=30000)
        self.rgb_encoder = TorchVisionResNet50().eval()
        self.depth_encoder = ResnetDepthEncoder().eval()
        self.map_encoder = MapEncoder(map_size)

        rnn_input_size = (
            self.instruction_encoder.output_size
            + self.depth_encoder.out_features
            + self.rgb_encoder.out_features
            + self.map_encoder.out_features
        )

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=self.state_encoder_hidden_size,
            rnn_type=state_encoder_rnn_type,
            num_layers=1,
        )

        self.goal_prediction_head = self.prediction_head = nn.Sequential(
            nn.Linear(self.state_encoder.hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Sigmoid(),
        )

        self.progress_prediction_head = self.prediction_head = nn.Sequential(
            nn.Linear(self.state_encoder.hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

        self.train()

    @property
    def output_size(self):
        return self.state_encoder_hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(
        self,
        tokenized_instruction_batch,
        depth_batch,
        rgb_batch,
        map_batch,
        rnn_states_batch,
        masks
    ):
        instruction_embedding = self.instruction_encoder(tokenized_instruction_batch)
        depth_embedding = self.depth_encoder(depth_batch)
        rgb_embedding = self.rgb_encoder(rgb_batch)
        map_embedding = self.map_encoder(map_batch)

        x = torch.cat(
            [instruction_embedding, depth_embedding, rgb_embedding, map_embedding], dim=1
        )

        x, rnn_states_out = self.state_encoder(x, rnn_states_batch, masks)

        pred_normalized_goal_xys = self.goal_prediction_head(x)
        pred_progress = self.progress_prediction_head(x)

        return pred_normalized_goal_xys, pred_progress, rnn_states_out
    
    def get_initial_recurrent_hidden_states(self, batch_size: int, device: str):
        return torch.zeros(
            batch_size,
            self.num_recurrent_layers,
            self.state_encoder.hidden_size,
            device=device
        )


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

    def forward(self, instruction) -> Tensor:
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        instruction = instruction.long()
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
