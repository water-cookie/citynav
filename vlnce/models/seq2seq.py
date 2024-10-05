import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import Space

from vlnce.aux_losses import AuxLosses
from vlnce.defaultpaths import DEPTH_ENCODER_WEIGHT_PATH

from .encoders.instruction_encoder import InstructionEncoder
from .encoders.resnet_encoders import TorchVisionResNet50, VlnResnetDepthEncoder
from .encoders.rnn_state_encoder import build_rnn_state_encoder


class Seq2SeqNet(nn.Module):
    """A baseline sequence to sequence network that performs single modality
    encoding of the instruction, RGB, and depth observations. These encodings
    are concatentated and fed to an RNN. Finally, a distribution over discrete
    actions (FWD, L, R, STOP) is produced.
    """

    def __init__(
        self,
        observation_space: Space,
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
            final_state_only=True,
            bidirectional=False,
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
        )

        # Init the RGB visual encoder
        self.rgb_encoder = TorchVisionResNet50(
            output_size=rgb_encoder_output_size,
            normalize_visual_inputs=False,
            trainable=False,
            spatial_output=False,
        )

        if self.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        # Init the RNN state decoder
        rnn_input_size = (
            self.instruction_encoder.output_size
            + depth_encoder_output_size
            + rgb_encoder_output_size
        )

        if self.use_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=self.state_encoder_hidden_size,
            rnn_type=state_encoder_rnn_type,
            num_layers=1,
        )

        self.progress_monitor = nn.Linear(
            self.state_encoder_hidden_size, 1
        )

        self._init_layers()

        self.train()

    @property
    def output_size(self):
        return self.state_encoder_hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _init_layers(self):
        nn.init.kaiming_normal_(
            self.progress_monitor.weight, nonlinearity="tanh"
        )
        nn.init.constant_(self.progress_monitor.bias, 0)

    def forward(self, observations, rnn_states, prev_actions, masks):
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)

        if self.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        x = torch.cat(
            [instruction_embedding, depth_embedding, rgb_embedding], dim=1
        )

        if self.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)

        x, rnn_states_out = self.state_encoder(x, rnn_states, masks)

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
    

# model_config = Config({'policy_name': 'Seq2SeqPolicy', 'normalize_rgb': False, 'ablate_depth': False, 'ablate_rgb': False, 'ablate_instruction': False, 'INSTRUCTION_ENCODER': Config({'sensor_uuid': 'instruction', 'vocab_size': 2504, 'use_pretrained_embeddings': True, 'embedding_file': '../data/vln_ce/R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz', 'dataset_vocab': '../data/vln_ce/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz', 'fine_tune_embeddings': False, 'embedding_size': 50, 'hidden_size': 128, 'rnn_type': 'LSTM', 'final_state_only': True, 'bidirectional': False}), 'RGB_ENCODER': Config({'cnn_type': 'TorchVisionResNet50', 'output_size': 256, 'trainable': False}), 'DEPTH_ENCODER': Config({'cnn_type': 'VlnResnetDepthEncoder', 'output_size': 128, 'backbone': 'resnet50', 'ddppo_checkpoint': '../data/vln_ce/ddppo-models/gibson-2plus-resnet50.pth', 'trainable': False}), 'STATE_ENCODER': Config({'hidden_size': 512, 'rnn_type': 'GRU'}), 'PROGRESS_MONITOR': Config({'use': False, 'alpha': 1.0}), 'SEQ2SEQ': Config({'use_prev_action': False}), 'WAYPOINT': Config({'predict_distance': True, 'continuous_distance': True, 'min_distance_var': 0.0625, 'max_distance_var': 3.52, 'max_distance_prediction': 2.75, 'min_distance_prediction': 0.25, 'discrete_distances': 6, 'predict_offset': True, 'continuous_offset': True, 'min_offset_var': 0.011, 'max_offset_var': 0.0685, 'discrete_offsets': 7, 'offset_temperature': 1.0})})