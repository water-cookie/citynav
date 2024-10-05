from typing import Optional

import numpy as np

from gsamllavanav.observation import cropclient
from gsamllavanav.defaultpaths import GSAM_MAPS_DIR
from gsamllavanav.space import Point2D, Pose4D
from gsamllavanav.dataset.episode import Episode

from .map import Map
from .tracking_map import TrackingMap
from .landmark_map import LandmarkMap
from .gsam_map import GSamMap, GSamParams


class LandmarkNavMap(Map):
    def __init__(
        self,
        map_name: str,
        map_shape: tuple[int, int],
        map_pixels_per_meter: float,
        landmark_names: list[str],
        target_name: str, surroundings_names: list[str],
        gsam_params: GSamParams,
    ):
        super().__init__(map_name, map_shape, map_pixels_per_meter)

        self.tracking_map = TrackingMap(map_name, map_shape, map_pixels_per_meter)
        self.landmark_map = LandmarkMap(map_name, map_shape, map_pixels_per_meter, landmark_names)
        self.target_map = GSamMap(map_name, map_shape, map_pixels_per_meter, [target_name], gsam_params)
        self.surroundings_map = GSamMap(map_name, map_shape, map_pixels_per_meter, surroundings_names, gsam_params)
    
    def update_observations(
        self,
        camera_pose: Pose4D,
        rgb: np.ndarray,
        depth_perspective: Optional[np.ndarray] = None,
        use_gsam_map_cache=True,
    ):
        self.tracking_map.mark_current_view_area(camera_pose)
        if use_gsam_map_cache:
            self.target_map.update_from_map_cache(camera_pose)
            self.surroundings_map.update_from_map_cache(camera_pose)
        else:
            self.target_map.update_observation(camera_pose, rgb[..., ::-1], depth_perspective)
            self.surroundings_map.update_observation(camera_pose, rgb[..., ::-1], depth_perspective)

    def to_array(self, dtype=np.float32) -> np.ndarray:
        return np.concatenate([
            self.tracking_map.to_array(dtype),
            self.landmark_map.to_array(dtype),
            self.target_map.to_array(dtype),
            self.surroundings_map.to_array(dtype),
        ])

    @classmethod
    def generate_maps_for_an_episode(
        cls,
        episode: Episode,
        map_shape: tuple[int, int],
        pixels_per_meter: float,
        update_interval: int,
        image_shape: tuple[int, int],
        gsam_params: GSamParams,
        use_gsam_map_cache=True,
    ):
        trajectory = episode.sample_trajectory(update_interval)

        # tracking map
        tracking_map = TrackingMap(episode.map_name, map_shape, pixels_per_meter)
        tracking_maps = np.stack([tracking_map.mark_current_view_area(pose).to_array() for pose in trajectory])
        assert tracking_maps.shape == (len(trajectory), 2, *map_shape)

        # landmark maps
        landmark_map = LandmarkMap(episode.map_name, map_shape, pixels_per_meter, episode.target_processed_description.landmarks)
        landmark_maps = np.tile(landmark_map.to_array(), (len(trajectory), 1, 1, 1))
        assert landmark_maps.shape == (len(trajectory), 1, *map_shape)

        # target & object maps
        target_map = GSamMap(episode.map_name, map_shape, pixels_per_meter, [episode.target_processed_description.target], gsam_params)
        surrounding_map = GSamMap(episode.map_name, map_shape, pixels_per_meter, episode.target_processed_description.surroundings, gsam_params)
        
        if use_gsam_map_cache:
            target_maps = np.stack([target_map.update_from_map_cache(pose).to_array() for pose in trajectory])
            surrounding_maps = np.stack([surrounding_map.update_from_map_cache(pose).to_array() for pose in trajectory])
        else:
            cropclient.load_image_cache()
            bgrs = [cropclient.crop_image(episode.map_name, pose, image_shape, 'rgb')[..., ::-1] for pose in trajectory]
            target_maps = np.stack([target_map.update_observation(pose, bgr).to_array() for pose, bgr in zip(trajectory, bgrs)])
            surrounding_maps = np.stack([surrounding_map.update_observation(pose, bgr).to_array() for pose, bgr in zip(trajectory, bgrs)])
        
        gsam_maps = np.concatenate((target_maps, surrounding_maps), axis=1)
        assert gsam_maps.shape == (len(trajectory), 2, *map_shape)

        episode_maps = np.concatenate((tracking_maps, landmark_maps, gsam_maps), axis=1)
        assert episode_maps.shape == (len(trajectory), 5, *map_shape)
        return episode_maps
    
    @classmethod
    def from_array(
        cls,
        map_name: str,
        map_shape: tuple[int, int],
        map_pixels_per_meter: float,
        landmark_names: list[str],
        target_name: str,
        object_names: list[str],
        map_data: np.ndarray,
    ):
        nav_map = LandmarkNavMap(map_name, map_shape, map_pixels_per_meter, landmark_names, target_name, object_names)
        nav_map.tracking_map.current_view_area = map_data[0].astype(np.uint8)
        nav_map.tracking_map.explored_area = map_data[1].astype(np.uint8)
        nav_map.landmark_map.landmark_map = map_data[2].astype(np.uint8)
        nav_map.target_map.gsam_map = map_data[3]
        nav_map.surroundings_map.gsam_map = map_data[4]

        return nav_map
    
    def plot(
        self,
        goal_description: str,
        predicted_goal: Point2D,
        true_goal: Point2D,
        show=False,
    ):
        import cv2

        predicted_goal_map = cv2.circle(
            img=np.zeros(self.shape, dtype=np.float32),
            center=self.to_row_col(predicted_goal)[::-1],
            radius=4, color=1, thickness=-1
        )

        true_goal_map = cv2.circle(
            img=np.zeros(self.shape, dtype=np.float32),
            center=self.to_row_col(true_goal)[::-1],
            radius=4, color=1, thickness=-1
        )

        titles = ['current view area', 'explored area', 'landmarks', 'target', 'surroundings', 'predicted goal', 'true goal']
        maps = np.concatenate([self.to_array(), np.stack([predicted_goal_map, true_goal_map])])

        import matplotlib.pyplot as plt
        from PIL import Image

        fig, axs = plt.subplots(nrows=1, ncols=7, figsize=(35, 5), subplot_kw={'xticks': [], 'yticks': []})
        fig.suptitle(f"{self.name}: {goal_description}")

        for ax, title, m in zip(axs, titles, maps):
            ax.imshow(m, cmap='viridis')
            ax.set_title(title)
        
        plt.tight_layout()
        fig.canvas.draw()

        if show:
            plt.show()
        
        plot_img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

        plt.close(fig)
        
        return plot_img