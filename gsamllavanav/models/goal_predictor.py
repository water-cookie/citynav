from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .ddppo.resenet_encoders import TorchVisionResNet50, ResnetDepthEncoder


class MapEncoder(nn.Module):
    '''Encodes maps of size (240, 240, 5) into a (15 * 15 * 32) feature vector'''

    def __init__(self, map_size: int):
        super(MapEncoder, self).__init__()

        self.main = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(5, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten()
        )

        self.out_features = (map_size // 2**4)**2 * 32
    
    def forward(self, maps):
        x = self.main(maps)
        return x


class GoalPredictionHead(nn.Module):

    def __init__(self, n_map_features: int):
        super(GoalPredictionHead, self).__init__()
        
        self.prediction_head = nn.Sequential(
            nn.Linear(n_map_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

    def forward(self, map_features):
        return self.prediction_head(map_features)


class ProgressPredictionHead(nn.Module):

    def __init__(self, n_map_features: int, n_rgb_features: int, n_depth_featuers):
        super(ProgressPredictionHead, self).__init__()

        self.prediction_head = nn.Sequential(
            nn.Linear(n_map_features + n_rgb_features + n_depth_featuers, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, map_features, rgb_features, depth_features):
        return self.prediction_head(torch.cat((map_features, rgb_features, depth_features), dim=1))
    

class GoalPredictor(nn.Module):

    def __init__(self, map_size: int):
        super(GoalPredictor, self).__init__()

        self.map_encoder = MapEncoder(map_size)
        self.rgb_encoder = TorchVisionResNet50().eval()
        self.depth_encoder = ResnetDepthEncoder().eval()

        self.goal_prediction_head = GoalPredictionHead(self.map_encoder.out_features)
        self.progress_prediction_head = ProgressPredictionHead(
            self.map_encoder.out_features, self.rgb_encoder.out_features, self.depth_encoder.out_features
        )
    
    def forward(self, maps: Tensor, rgbs: Tensor, depths: Tensor, flip_depth=True):
        """rgb & depth (B, C, H, W)"""

        if flip_depth:
            depths = depths.flip(-2)  # flip vertically

        map_features = self.map_encoder(maps)
        rgb_features = self.rgb_encoder(rgbs)
        depth_features = self.depth_encoder(depths)
        
        pred_normalized_goal_xys = self.goal_prediction_head(map_features)
        pred_progress = self.progress_prediction_head(map_features, rgb_features, depth_features)

        return pred_normalized_goal_xys, pred_progress

    def predict(
        self,
        to_world_xy: Callable[[tuple[float, float]], tuple[float, float]],
        maps: Tensor, rgb: Tensor, depth: Tensor,
        flip_depth=True
    ):
        pred_normalized_goal_coords, pred_progress = self(maps, rgb, depth, flip_depth)
        pred_xy = to_world_xy(pred_normalized_goal_coords)
        return pred_xy, pred_progress