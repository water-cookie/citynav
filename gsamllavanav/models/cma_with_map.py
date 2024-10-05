from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import resnet50 as torchvision_resnet50
from torchvision.transforms.v2 import Normalize

from vlnce.defaultpaths import DEPTH_ENCODER_WEIGHT_PATH
from vlnce.models.encoders.rnn_state_encoder import build_rnn_state_encoder

from .ddppo.resenet_encoders import ResNetEncoder
from .goal_predictor import MapEncoder
from .seq2seq_with_map import InstructionEncoder


class CMAwithMap(nn.Module):

    def __init__(
        self,
        map_size: int,
        depth_encoder_output_size=128,
        rgb_encoder_output_size=256,
        hidden_size=512,
        rnn_type='GRU',
    ):
        super().__init__()

        self._output_size = self._hidden_size = hidden_size

        # input encoders
        self.instruction_encoder = InstructionEncoder(final_state_only=False, bidirectional=True, vocab_size=30000)
        self.depth_encoder = VlnResnetDepthEncoder().eval()
        self.rgb_encoder = TorchVisionResNet50().eval()
        self.map_encoder = MapEncoder(map_size)
        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.rgb_encoder.output_shape[0], rgb_encoder_output_size),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.depth_encoder.output_shape), depth_encoder_output_size,),
            nn.ReLU(True),
        )

        # first GRU
        self.first_state_encoder = build_rnn_state_encoder(
            input_size=depth_encoder_output_size + rgb_encoder_output_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=1,
        )

        # first attention
        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(self.instruction_encoder.output_size, hidden_size // 2, 1)

        # second attention
        self.text_q = nn.Linear(self.instruction_encoder.output_size, hidden_size // 2)
        self.rgb_kv = nn.Conv1d(self.rgb_encoder.output_shape[0], hidden_size // 2 + rgb_encoder_output_size, 1)
        self.depth_kv = nn.Conv1d(self.depth_encoder.output_shape[0], hidden_size // 2 + depth_encoder_output_size, 1)

        # second GRU
        self.second_state_compress = nn.Sequential(
            nn.Linear(
                (
                    hidden_size
                    + rgb_encoder_output_size
                    + depth_encoder_output_size
                    + self.instruction_encoder.output_size
                    + self.map_encoder.out_features
                ), 
                hidden_size
            ),
            nn.ReLU(True),
        )
        self.second_state_encoder = build_rnn_state_encoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=1,
        )

        # prediction heads
        self.goal_prediction_head = self.prediction_head = nn.Sequential(
            nn.Linear(self.second_state_encoder.hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Sigmoid(),
        )
        self.progress_prediction_head = self.prediction_head = nn.Sequential(
            nn.Linear(self.second_state_encoder.hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

        self.register_buffer("_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5)))
        self.train()


    @property
    def state_encoder_hidden_size(self):
        return self._hidden_size

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        return self.first_state_encoder.num_recurrent_layers + self.second_state_encoder.num_recurrent_layers


    def _attn(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(
        self,
        tokenized_instruction_batch,
        depth_batch,
        rgb_batch,
        map_batch,
        rnn_states_batch: torch.Tensor,
        masks
    ):
        n_layers = self.first_state_encoder.num_recurrent_layers
        first_hidden = rnn_states_batch[:, :n_layers]
        second_hidden = rnn_states_batch[:, n_layers:]

        instruction_embedding = self.instruction_encoder(tokenized_instruction_batch)
        encoded_depth = self.depth_encoder(depth_batch).flatten(2)
        encoded_rgb = self.rgb_encoder(rgb_batch).flatten(2)
        depth_features = self.depth_linear(encoded_depth)
        rgb_features = self.rgb_linear(encoded_rgb)
        map_features = self.map_encoder(map_batch)

        # 1st GRU
        state_in = torch.cat([rgb_features, depth_features], dim=1)
        x, first_hidden = self.first_state_encoder(state_in, first_hidden, masks)

        # rnn encoded rgbd <-> instruction cross attention
        x_q = self.state_q(x)
        text_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        attended_text = self._attn(x_q, text_k, instruction_embedding, text_mask)

        # attended instruction <-> rgbd cross attention
        text_q = self.text_q(attended_text)
        rgb_k, rgb_v = torch.split(self.rgb_kv(encoded_rgb), self._hidden_size // 2, dim=1)
        depth_k, depth_v = torch.split(self.depth_kv(encoded_depth), self._hidden_size // 2, dim=1)
        attended_rgb = self._attn(text_q, rgb_k, rgb_v)
        attended_depth = self._attn(text_q, depth_k, depth_v)

        # 2nd GRU
        x = torch.cat([x, attended_text, attended_rgb, attended_depth, map_features], dim=1)
        x = self.second_state_compress(x)
        x, second_hidden = self.second_state_encoder(x, second_hidden, masks)

        # prediction heads
        pred_normalized_goal_xys = self.goal_prediction_head(x)
        pred_progress = self.progress_prediction_head(x)

        return pred_normalized_goal_xys, pred_progress, torch.cat((first_hidden, second_hidden), dim=1)
    
    def get_initial_recurrent_hidden_states(self, batch_size: int, device: str):
        return torch.zeros(
            batch_size,
            self.num_recurrent_layers,
            self.state_encoder_hidden_size,
            device=device
        )


class VlnResnetDepthEncoder(nn.Module):

    def __init__(self, checkpoint=DEPTH_ENCODER_WEIGHT_PATH):
        super().__init__()

        self.visual_encoder = ResNetEncoder()
        self.visual_encoder.load_checkpoint(checkpoint)

        for param in self.visual_encoder.parameters():
            param.requires_grad_(False)

        out_c, out_h, out_w = self.visual_encoder.output_shape

        self.spatial_embeddings = nn.Embedding(out_h * out_w, 64)

        self.output_shape = (out_c + 64, out_h, out_w)

    def forward(self, depth: Tensor) -> Tensor:

        x = self.visual_encoder(depth)

        b, c, h, w = x.size()

        indicies = torch.arange(0, self.spatial_embeddings.num_embeddings, device=x.device, dtype=torch.long)
        spatial_features = self.spatial_embeddings(indicies).view(1, -1, h, w).expand(b, self.spatial_embeddings.embedding_dim, h, w)

        return torch.cat([x, spatial_features], dim=1)


class TorchVisionResNet50(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)

        resnet = torchvision_resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        class SpatialAvgPool(nn.Module):
            def forward(self, x):
                x = F.adaptive_avg_pool2d(x, (4, 4))
                return x
        self.cnn.avgpool = SpatialAvgPool()

        for param in self.cnn.parameters():
            param.requires_grad_(False)
        self.cnn.train(False)

        self.spatial_embeddings = nn.Embedding(4 * 4, 64)

        self.output_shape = (resnet.fc.in_features + 64, 4, 4)

    def forward(self, x: Tensor) -> Tensor:

        x = x.contiguous() / 255.
        x = self.normalize(x)
        resnet_output = self.cnn(x)

        b, c, h, w = resnet_output.size()
        n_embeddings = self.spatial_embeddings.num_embeddings
        
        indicies = torch.arange(0, self.spatial_embeddings.num_embeddings, device=x.device, dtype=torch.long)
        spatial_features = self.spatial_embeddings(indicies).view(1, -1, h, w).expand(b, self.spatial_embeddings.embedding_dim, h, w)

        return torch.cat([resnet_output, spatial_features], dim=1)