import json
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

from gsamllavanav import som
from gsamllavanav.dataset.episode import Episode
from gsamllavanav.defaultpaths import SUBBLOCKS_DIR
from gsamllavanav.mapdata import GROUND_LEVEL, MAP_BOUNDS
from gsamllavanav.observation import cropclient
from gsamllavanav.space import Point2D, Point3D, Pose4D, crwh_to_global_bbox, bbox_corners_to_position


@dataclass
class SubBlock:
    map_name: str
    pose: Pose4D
    annotated_rgb: np.ndarray
    masks: list[dict]
    segmentation_model: str
    level: list[int]

    @property
    def shape(self):
        return self.annotated_rgb.shape[:2]
    
    @property
    def ground_level(self):
        return GROUND_LEVEL[self.map_name]
    
    @property
    def altitude_from_ground(self):
        return self.pose.z - self.ground_level

    @property
    def labels(self):
        """labels start from 1 not 0"""
        return set(range(1, len(self.masks) + 1))

    def bbox(self, label: int):
        return crwh_to_global_bbox(self.masks[label - 1]['crwh'], self.shape, self.pose, self.ground_level)
    
    def bbox_pos(self, label: int):
        return bbox_corners_to_position(self.bbox(label), self.ground_level)
    
    def contains(self, position: Point2D | Point3D | Pose4D):

        stride = self.altitude_from_ground
        x_min = self.pose.x - stride
        x_max = self.pose.x + stride
        y_min = self.pose.y - stride
        y_max = self.pose.y + stride
        
        return x_min < position.x < x_max and y_min < position.y < y_max


SUBBLOCKS_ID = tuple[str, float, str]
_subblocks_cache: dict[SUBBLOCKS_ID, list[SubBlock]] = dict()


def from_episode(
    episode: Episode,
    model_name: som.ModelName = 'semantic-sam',
    level: list[int] = [4],
    image_size: tuple[int, int] = (500, 500)
):
    
    map_name = episode.map_name
    altitude_from_ground = {'Building': 50, 'Car': 30, 'Ground': 70, 'Parking': 70}[episode.target_type]

    return load(map_name, altitude_from_ground, model_name, level, image_size)


def load(
    map_name: str,
    altitude_from_ground: float,
    model_name: som.ModelName,
    level: list[int],
    image_size: tuple[int, int] = (500, 500),
) -> list[SubBlock]:
    
    global _subblocks_cache
    
    model_level = f"{model_name}_{sorted(level)}" if model_name == 'semantic-sam' else model_name
    subblocks_id = (map_name, altitude_from_ground, model_level)

    if subblocks_id not in _subblocks_cache:
        _subblocks_cache[subblocks_id] = _generate_subblocks(map_name, altitude_from_ground, image_size, model_name, level)
    
    return _subblocks_cache[subblocks_id]


def unload_model(model_name: som.ModelName):
    som.unload_model(model_name)


def delete_cache():
    del _subblocks_cache


def _generate_subblocks(
    map_name: str,
    altitude_from_ground: float,
    image_size: tuple[int, int],
    model_name: som.ModelName,
    level : list[int] = [],
    save=True
) -> list[SubBlock]:
    
    # dir name
    model = f"{model_name}_{sorted(level)}" if model_name == "semantic-sam" else model_name
    map_dir = SUBBLOCKS_DIR/model/str(altitude_from_ground)/map_name
    map_dir.mkdir(parents=True, exist_ok=True)

    poses = _split_map(map_name, altitude_from_ground)
    
    subblocks = []
    for pose in tqdm(poses, desc="annotating_images", unit='image', leave=False, position=1):

        # file name
        filename = f"{tuple(pose)}_{image_size}"
        rgb_path = map_dir/f"{filename}.png"
        mask_path = map_dir/f"{filename}.json"

        if rgb_path.exists():
            annotated_rgb = cv2.imread(str(rgb_path))[..., ::-1]
            with open(mask_path) as f:
                masks = json.load(f)
        else:
            cropclient.load_image_cache()
            rgb = cropclient.crop_image(map_name, pose, image_size, 'rgb')
            annotated_rgb, masks = som.annotate(rgb, model_name, level)
            masks = [
                {'label_id': i + 1, 'area': mask['area'], 'crwh': mask['bbox']} 
                for i, mask in enumerate(masks)
            ]
            if save:
                cv2.imwrite(str(rgb_path), annotated_rgb[..., ::-1])
                with open(mask_path, 'w') as f:
                    json.dump(masks, f)
        
        
        subblocks.append(SubBlock(map_name, pose, annotated_rgb, masks, model_name, level))
    
    return subblocks


def _split_map(map_name: str, altitude_from_ground: float):

    x_min, y_min, x_max, y_max = MAP_BOUNDS[map_name]
    z = altitude_from_ground + GROUND_LEVEL[map_name]
    stride = altitude_from_ground

    x_start = x_min + stride if x_min + stride < x_max else (x_min + x_max) / 2
    y_start = y_min + stride if y_min + stride < y_max else (y_min + y_max) / 2

    pose_subblocks = [
        Pose4D(x, y, z, 0)
        for x in np.arange(x_start, x_max, stride)
        for y in np.arange(y_start, y_max, stride)
    ]

    return pose_subblocks
