import numpy as np

from gsamllavanav.mapdata import MAP_BOUNDS, GROUND_LEVEL
from gsamllavanav.space import Point2D


class Map:

    def __init__(self, map_name: str, shape: tuple[int, int], pixels_per_meter: float):
        self.name = map_name
        self.shape = shape
        self.pixels_per_meter = pixels_per_meter
    
    @property
    def bounds(self):
        return MAP_BOUNDS[self.name]
    
    @property
    def ground_level(self):
        return GROUND_LEVEL[self.name]
    
    @property
    def size_meters(self):
        return self.shape[0] / self.pixels_per_meter
    
    def to_row_col(self, world_xy: Point2D) -> tuple[int, int]:
        x, y = world_xy
        x_min, y_min, x_max, y_max = self.bounds

        col = round((x - x_min) * self.pixels_per_meter)
        row = round((y_max - y) * self.pixels_per_meter)

        return row, col
    
    def to_rows_cols(self, world_xys: list[Point2D]) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.array(world_xys).T
        x_min, y_min, x_max, y_max = self.bounds

        col = np.round((x - x_min) * self.pixels_per_meter).astype(int)
        row = np.round((y_max - y) * self.pixels_per_meter).astype(int)

        return row, col
    
    def to_world_xy(self, row: int, col: int) -> Point2D:
        x_min, y_min, x_max, y_max = self.bounds

        x = x_min + col / self.pixels_per_meter
        y = y_max - row / self.pixels_per_meter

        return Point2D(x, y)

    def to_world_xys(self, rows: np.ndarray, cols: np.ndarray) -> list[Point2D]:
        x_min, y_min, x_max, y_max = self.bounds

        xs = x_min + cols / self.pixels_per_meter
        ys = y_max - rows / self.pixels_per_meter
        xys = [Point2D(x, y) for x, y in zip(xs, ys)]

        return xys

    def view_radius_pixels(self, z: float) -> int:
        altitude_from_ground = z - self.ground_level
        return round(altitude_from_ground * self.pixels_per_meter)