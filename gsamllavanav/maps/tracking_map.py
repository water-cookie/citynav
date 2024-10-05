import cv2
import numpy as np

from gsamllavanav.space import Pose4D, view_area_corners

from .map import Map


class TrackingMap(Map):

    def __init__(self, map_name: str, map_shape: tuple[int, int], pixels_per_meter: float):
        super().__init__(map_name, map_shape, pixels_per_meter)
        
        self.current_view_area = np.zeros(map_shape, dtype=np.uint8)
        self.explored_area = np.zeros(map_shape, dtype=np.uint8)

    def mark_current_view_area(self, pose: Pose4D):
        # mark on maps
        rows, cols = self.to_rows_cols(view_area_corners(pose, self.ground_level))
        view_corners = np.stack((cols, rows)).T

        self.current_view_area = cv2.fillConvexPoly(np.zeros(self.shape, dtype=np.uint8), view_corners, color=1)
        self.explored_area = np.maximum(self.explored_area, self.current_view_area)
        return self

    def to_array(self, dtype=np.float32) -> np.ndarray:
        return np.stack([
            self.current_view_area,
            self.explored_area
        ]).astype(dtype)
