import cv2
import numpy as np
import Levenshtein

from gsamllavanav.cityreferobject import get_landmarks, remove_duplicate_landmarks_by_area

from .map import Map


class LandmarkMap(Map):
    _landmarks_cache = None
    _landmark_segmentations = None

    def __init__(
        self,
        map_name: str,
        map_shape: tuple[int, int],
        pixels_per_meter: float,
        landmark_names: list[str],
    ):
        super().__init__(map_name, map_shape, pixels_per_meter)
        self.landmark_names = landmark_names
        self.landmarks = LandmarkMap._search_landmarks_by_name(map_name, landmark_names)

        self.landmark_map = np.zeros(map_shape, dtype=np.uint8)
        for lm in self.landmarks:
            self.landmark_map = cv2.fillPoly(
                img=self.landmark_map ,
                pts=[np.stack(self.to_rows_cols(lm.contour))[::-1].T],
                color=1
            )
    
    def to_array(self, dtype=np.float32) -> np.ndarray:
        return self.landmark_map[np.newaxis].astype(dtype)
    
    @classmethod
    def _search_landmarks_by_name(cls, map_name: str, query_names: list[str]):
        # load landmark data
        if cls._landmarks_cache is None:
            cls._landmarks_cache = remove_duplicate_landmarks_by_area(get_landmarks())

        landmarks = cls._landmarks_cache[map_name].values()

        return [
            min(landmarks, key=lambda lm, q=query: Levenshtein.distance(lm.name, q))
            for query in query_names
        ]
