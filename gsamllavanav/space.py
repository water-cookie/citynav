from __future__ import annotations

from math import pi, ceil, floor
from typing import NamedTuple

import numpy as np
from shapely.geometry import Polygon


class Point2D(NamedTuple):
    x: float
    y: float

    def dist_to(self, other: Point2D):
        return np.linalg.norm(np.array(self) - np.array(other))


class Point3D(NamedTuple):
    x: float
    y: float
    z: float

    @property
    def xy(self):
        return Point2D(self.x, self.y)

    def dist_to(self, other: Point3D):
        return np.linalg.norm(np.array(self) - np.array(other))


class Pose4D(NamedTuple):
    x: float
    y: float
    z: float
    yaw: float

    @property
    def xyz(self):
        return Point3D(self.x, self.y, self.z)
    
    @property
    def xy(self):
        return Point2D(self.x, self.y)
    

class Pose5D(NamedTuple):
    x: float
    y: float
    z: float
    yaw: float
    pitch: float

    @classmethod
    def from_direction_vector(cls, x: float, y: float, z: float, dx: float, dy: float, dz: float):
        yaw = np.arctan2(dy, dx)
        pitch = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
        return Pose5D(x, y, z, yaw, pitch)
    
    @property
    def xyzyaw(self):
        return Pose4D(self.x, self.y, self.z, self.yaw)
    
    @property
    def xyz(self):
        return Point3D(self.x, self.y, self.z)


def bbox_corners_to_position(corners: list[Point2D], ground_level: float):
    
    corners = np.array(corners)
    x, y = corners.mean(axis=0)
    
    box_width = np.linalg.norm(corners[0] - corners[1])
    box_height = np.linalg.norm(corners[0] - corners[-1])
    z = max(box_width, box_height) + ground_level

    return Point3D(x, y, z)


def bbox_IoU(bbox1: list[Point2D], bbox2: list[Point2D]) -> float:
    
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    return intersection / union


def crwh_to_global_bbox(
    crwh: tuple[int, int, int, int],
    image_size: tuple[int, int],
    pose: Pose4D,
    ground_level: float
):
    """converts bbox of format (column, row, width, height) to global xy coordinates"""
    c, r, w, h = crwh
    
    x1 = c - w / 2
    x2 = c + w / 2
    y1 = r - h / 2
    y2 = r + w / 2
    
    return xyxy_to_global_bbox((x1, y1, x2, y2), image_size, pose, ground_level)


def xyxy_to_global_bbox(
    xyxy: tuple[float, float, float, float],
    image_size: tuple[int, int],
    pose: Pose4D,
    ground_level: float
):
    x1, y1, x2, y2 = xyxy

    bbox_corners_col_row = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ])

    n_rows, n_cols =  image_size

    cos, sin = np.cos(pose.yaw), np.sin(pose.yaw)
    front = np.array([cos, sin])
    left = np.array([-sin, cos])
    center = np.array([pose.x, pose.y])
    view_area_size = pose.z - ground_level
    
    xy_per_row = - (view_area_size / n_rows) * front
    xy_per_col = - (view_area_size / n_cols) * left

    bbox_corners_xy = center + bbox_corners_col_row @ np.stack([xy_per_row, xy_per_col])
    bbox_corners_xy = [Point2D(x, y) for x, y in bbox_corners_xy]

    return bbox_corners_xy


def modulo_radians(theta: float):
    ''' projects radians to range [-pi, pi) '''
    return (theta + pi) % (2*pi) - pi


def view_area_corners(pose: Pose4D, ground_level: float):
    
    cos, sin = np.cos(pose.yaw), np.sin(pose.yaw)
    front = np.array([cos, sin])
    left = np.array([-sin, cos])
    center = np.array([pose.x, pose.y])

    # compute view area corners
    altitude_from_ground = pose.z - ground_level
    view_area_corners_xy = [
        center + altitude_from_ground * (front + left),
        center + altitude_from_ground * (front - left),  # front right
        center + altitude_from_ground * (-front - left),  # back right
        center + altitude_from_ground * (-front + left),  # back left
    ]

    return [Point2D(x, y) for x, y in view_area_corners_xy]