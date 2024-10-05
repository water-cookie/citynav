from typing import Literal, Callable

import numpy as np

from gsamllavanav.space import Point3D


TrajectoryType = Literal['linear', 'linear_xy', 'move_and_drop']

def straight_line_trajectory(src_pos: Point3D, dst_pos: Point3D, stride_meters=5):
    '''returns a trajectory connecting `src_pos` and `dst_pos` in a straight line
    
    Parameters
    ----------
    src_pos
        3D coordinates of the source position
    dst_pos
        3D coordinates of the destination position
    stride_meters
        Each step advances `stride_meters` towards the `dst_pos` except for the last step.
        The stride of the last step is `stride_meters` < last_stride < 2 * `stride_meters`.
    '''
    
    src_pos = np.array(src_pos)
    dst_pos = np.array(dst_pos)
    
    distance_to_dst = np.linalg.norm(dst_pos - src_pos)

    stride_vector = (dst_pos - src_pos) / (distance_to_dst + 0.01) * stride_meters
    num_steps = int(distance_to_dst / stride_meters)

    trajectory = [src_pos + i * stride_vector for i in range(num_steps)]
    trajectory.append(dst_pos)
    
    trajectory = [Point3D(*waypoint) for waypoint in trajectory]

    return trajectory


def planar_straight_line_trajectory(src_pos: Point3D, dst_pos: Point3D, stride_meters=5):
    dst_pos = Point3D(dst_pos.x, dst_pos.y, src_pos.z)
    return straight_line_trajectory(src_pos, dst_pos, stride_meters)


def move_and_drop_trajectory(src_pos: Point3D, dst_pos: Point3D, move_stride_meters=5, drop_meters=2):
    '''moves towards `dst_pos` without changing the altitude, then changes the altitiude to approach `dst_pos`'''
    
    via = Point3D(dst_pos.x, dst_pos.y, src_pos.z)
    traj_move = straight_line_trajectory(src_pos, via, move_stride_meters)[:-1]
    traj_drop = straight_line_trajectory(via, dst_pos, drop_meters)

    return traj_move + traj_drop


def trajectory_length(trajectory: list[Point3D]) -> float:
    '''the lengthof the trajectory in meters'''

    trajectory = np.array(trajectory)
    stride_sizes = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=-1)
    trajectory_length = stride_sizes.sum()
    return trajectory_length


trajectory_registry : dict[TrajectoryType, Callable[[Point3D, Point3D], list[Point3D]]] = {
    'linear': straight_line_trajectory,
    'linear_xy': planar_straight_line_trajectory,
    'move_and_drop': move_and_drop_trajectory,
}