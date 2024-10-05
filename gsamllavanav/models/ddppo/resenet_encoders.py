import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import resnet50 as torchvision_resnet50
from torchvision.transforms.v2 import Normalize

from gsamllavanav.defaultpaths import DEPTH_ENCODER_CHECKPOINT_PATH

from .resnet import resnet50


class ResNetEncoder(nn.Module):
    def __init__(self, input_channels=1, baseplanes=32, ngroups=16, spatial_size=128):  # spatial_size: half of image height or width
        super().__init__()
        
        self.backbone = resnet50(input_channels, baseplanes, ngroups)
        
        final_spatial = int(spatial_size * self.backbone.final_spatial_compress)
        num_compression_channels = int(round((after_compression_flat_size := 2048) / (final_spatial ** 2)))
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
        )

        self.output_shape = (num_compression_channels, final_spatial, final_spatial)

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, x: Tensor) -> Tensor:
        '''assumes x of shape (B, C, H, W)'''
        x = F.avg_pool2d(x, 2)
        x = self.backbone(x)
        x = self.compression(x)
        return x

    def load_checkpoint(self, checkpoint: str):
        ddppo_weights = torch.load(checkpoint)

        weights_dict = {}
        for k, v in ddppo_weights["state_dict"].items():
            split_layer_name = k.split(".")[2:]
            if split_layer_name[0] != "visual_encoder":
                continue

            layer_name = ".".join(split_layer_name[1:])
            weights_dict[layer_name] = v

        del ddppo_weights
        self.load_state_dict(weights_dict, strict=True)


class ResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        out_features=128,
        checkpoint=DEPTH_ENCODER_CHECKPOINT_PATH,
    ) -> None:
        super().__init__()
        
        self.visual_encoder = ResNetEncoder()
        self.visual_encoder.load_checkpoint(checkpoint)
        
        for param in self.visual_encoder.parameters():
            param.requires_grad_(False)

        out_c, out_h, out_w = self.visual_encoder.output_shape
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_c * out_h * out_w, out_features),
            nn.ReLU(True)
        )

        self.out_features = out_features

    def forward(self, depth: Tensor) -> Tensor:
        '''assumes depth of shape (B, C, H, W) & depth value of range (0-1) & flip vertically before input?'''
        depth = self.visual_encoder(depth)
        return self.fc(depth)



class TorchVisionResNet50(nn.Module):
    def __init__(self, out_features=256):
        super().__init__()

        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)

        # extract the CNN part from torchvision resnet 50
        resnet = torchvision_resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad_(False)
        self.cnn.train(False)

        self.fc = nn.Sequential(  # The final fc layer is replaced with a new fc layer of a specified output size.
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, out_features),
            nn.ReLU()
        )

        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        '''assumes x of shape (B, C, H, W)'''
        x = x.contiguous() / 255.
        x = self.normalize(x)
        x = self.cnn(x)
        x = self.fc(x)
        return x