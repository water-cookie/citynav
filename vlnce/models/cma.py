from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import Space
from torch import Tensor

from vlnce.aux_losses import AuxLosses
from vlnce.defaultpaths import DEPTH_ENCODER_WEIGHT_PATH

from .encoders.instruction_encoder import InstructionEncoder
from .encoders.resnet_encoders import TorchVisionResNet50, VlnResnetDepthEncoder
from .encoders.rnn_state_encoder import build_rnn_state_encoder


class CMANet(nn.Module):
    """An implementation of the cross-modal attention (CMA) network in https://arxiv.org/abs/2004.02857"""

    def __init__(
        self, observation_space: Space,
        num_actions: int,
        use_prev_action=False,
        depth_encoder_output_size=128,
        rgb_encoder_output_size=256,
        state_encoder_hidden_size=512,
        state_encoder_rnn_type='GRU',
        ablate_instruction=False,
        ablate_depth=False,
        ablate_rgb=False,
        use_progress_monitor=False,
        progress_monitor_alpha=1.0
    ):
        super().__init__()
        self.use_prev_action = use_prev_action
        self.state_encoder_hidden_size = state_encoder_hidden_size
        self.ablate_instruction = ablate_instruction
        self.ablate_depth = ablate_depth
        self.ablate_rgb = ablate_rgb
        self.use_progress_monitor = use_progress_monitor,
        self.progress_monitor_alpha = progress_monitor_alpha

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            embedding_size=50,
            hidden_size=128,
            rnn_type='LSTM',
            final_state_only=False,
            bidirectional=True,
            use_pretrained_embeddings=False,
            vocab_size=30000,
        )

        # Init the depth encoder
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=depth_encoder_output_size,
            checkpoint=str(DEPTH_ENCODER_WEIGHT_PATH),
            backbone='resnet50',
            trainable=False,
            spatial_output=True,
        )

        # Init the RGB visual encoder
        self.rgb_encoder = TorchVisionResNet50(
            output_size=rgb_encoder_output_size,
            normalize_visual_inputs=False,
            trainable=False,
            spatial_output=True,
        )

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        hidden_size = self.state_encoder_hidden_size
        self._hidden_size = hidden_size

        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.rgb_encoder.output_shape[0],
                rgb_encoder_output_size
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                depth_encoder_output_size,
            ),
            nn.ReLU(True),
        )

        # Init the RNN state decoder
        rnn_input_size = depth_encoder_output_size
        rnn_input_size += rgb_encoder_output_size
        rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=self.state_encoder_hidden_size,
            rnn_type=state_encoder_rnn_type,
            num_layers=1,
        )

        self._output_size = (
            self.state_encoder_hidden_size
            + rgb_encoder_output_size
            + depth_encoder_output_size
            + self.instruction_encoder.output_size
        )

        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            hidden_size // 2 + rgb_encoder_output_size,
            1,
        )

        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + depth_encoder_output_size,
            1,
        )

        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size, hidden_size // 2, 1
        )
        self.text_q = nn.Linear(
            self.instruction_encoder.output_size, hidden_size // 2
        )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        )

        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size + self.prev_action_embedding.embedding_dim,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=state_encoder_rnn_type,
            num_layers=1,
        )
        self._output_size = self.state_encoder_hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)

        self._init_layers()

        self.train()

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

    def _init_layers(self) -> None:
        if self.use_progress_monitor:
            nn.init.kaiming_normal_(
                self.progress_monitor.weight, nonlinearity="tanh"
            )
            nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        depth_embedding = torch.flatten(depth_embedding, 2)

        rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )

        if self.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        rgb_in = self.rgb_linear(rgb_embedding)
        depth_in = self.depth_linear(depth_embedding)

        state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
        rnn_states_out = rnn_states.detach().clone()
        
        (
            state,
            rnn_states_out[:, 0 : self.state_encoder.num_recurrent_layers],
        ) = self.state_encoder(
            state_in,
            rnn_states[:, 0 : self.state_encoder.num_recurrent_layers],
            masks,
        )

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )

        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1
        )

        text_q = self.text_q(text_embedding)
        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding = self._attn(text_q, depth_k, depth_v)

        x = torch.cat(
            [
                state,
                text_embedding,
                rgb_embedding,
                depth_embedding,
                prev_actions,
            ],
            dim=1,
        )
        x = self.second_state_compress(x)
        (
            x,
            rnn_states_out[:, self.state_encoder.num_recurrent_layers :],
        ) = self.second_state_encoder(
            x,
            rnn_states[:, self.state_encoder.num_recurrent_layers :],
            masks,
        )

        if self.use_progress_monitor and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1),
                observations["progress"],
                reduction="none",
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.progress_monitor_alpha,
            )

        return x, rnn_states_out

# model_config = Config({'policy_name': 'CMAPolicy', 'normalize_rgb': False, 'ablate_depth': False, 'ablate_rgb': False, 'ablate_instruction': False, 'INSTRUCTION_ENCODER': Config({'sensor_uuid': 'instruction', 'vocab_size': 2504, 'use_pretrained_embeddings': True, 'embedding_file': '../data/vln_ce/R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz', 'dataset_vocab': '../data/vln_ce/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz', 'fine_tune_embeddings': False, 'embedding_size': 50, 'hidden_size': 128, 'rnn_type': 'LSTM', 'final_state_only': True, 'bidirectional': True}), 'RGB_ENCODER': Config({'cnn_type': 'TorchVisionResNet50', 'output_size': 256, 'trainable': False}), 'DEPTH_ENCODER': Config({'cnn_type': 'VlnResnetDepthEncoder', 'output_size': 128, 'backbone': 'resnet50', 'ddppo_checkpoint': '../data/vln_ce/ddppo-models/gibson-2plus-resnet50.pth', 'trainable': False}), 'STATE_ENCODER': Config({'hidden_size': 512, 'rnn_type': 'GRU'}), 'PROGRESS_MONITOR': Config({'use': False, 'alpha': 1.0}), 'SEQ2SEQ': Config({'use_prev_action': False}), 'WAYPOINT': Config({'predict_distance': True, 'continuous_distance': True, 'min_distance_var': 0.0625, 'max_distance_var': 3.52, 'max_distance_prediction': 2.75, 'min_distance_prediction': 0.25, 'discrete_distances': 6, 'predict_offset': True, 'continuous_offset': True, 'min_offset_var': 0.011, 'max_offset_var': 0.0685, 'discrete_offsets': 7, 'offset_temperature': 1.0})})