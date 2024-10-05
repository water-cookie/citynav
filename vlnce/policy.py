from typing import Any

import torch
from gymnasium import spaces
from torch import nn
import numpy as np

from vlnce.models.distributions import CategoricalNet, CustomFixedCategorical
from vlnce.models.cma import CMANet
from vlnce.models.seq2seq import Seq2SeqNet
from vlnce.actions import DiscreteAction


def observation_space(rgb_size: tuple[int, int], depth_size: tuple[int, int]):
    return spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(*rgb_size, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0, high=1, shape=(*depth_size, 1), dtype=np.float32),
            "instruction": spaces.Sequence(spaces.Discrete(30000)),  # vocab size of bert-base-uncased
            "progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        }
    )

ACTION_SPACE = spaces.Discrete(len(DiscreteAction))


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        """Defines an imitation learning policy as having functions act() and
        build_distribution().
        """
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_states = self.net(
            observations, rnn_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        return action, rnn_states

    def get_value(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def evaluate_actions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def build_distribution(
        self, observations, rnn_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        features, rnn_states = self.net(
            observations, rnn_states, prev_actions, masks
        )
        return self.action_distribution(features)
    
    def get_initial_recurrent_hidden_states(self, batch_size: int, device: str):
        return torch.zeros(
            batch_size,
            self.net.num_recurrent_layers,
            self.net.state_encoder.hidden_size,
            device=device
        )
    
    def load_checkpoint(self, path: str):
        if path:
            checkpoint = torch.load(path)['policy_state_dict']
            self.load_state_dict(checkpoint)
        
        return self


class Seq2SeqPolicy(Policy):
    def __init__(
        self,
        rgb_size=(224, 224),
        depth_size=(256, 256),
        use_progress_monitor=False,
    ):
        super().__init__(
            Seq2SeqNet(
                observation_space=observation_space(rgb_size, depth_size),
                num_actions=ACTION_SPACE.n,
                use_progress_monitor=use_progress_monitor,
            ),
            ACTION_SPACE.n,
        )


class CMAPolicy(Policy):
    def __init__(
        self,
        rgb_size=(224, 224),
        depth_size=(256, 256),
        use_progress_monitor=False,
    ):
        super().__init__(
            CMANet(
                observation_space=observation_space(rgb_size, depth_size),
                num_actions=ACTION_SPACE.n,
                use_progress_monitor=use_progress_monitor,
            ),
            ACTION_SPACE.n,
        )

