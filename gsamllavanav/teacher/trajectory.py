from typing import Literal, Union

import numpy as np

from .algorithm.lookahead import LookaheadTeacherParams, lookahead_discrete_action
from gsamllavanav.actions import DiscreteAction
from gsamllavanav.space import Pose4D, Point3D, modulo_radians


DiscreteActionID = int
TeacherType = Literal['lookahead']
TeacherParams = Union[LookaheadTeacherParams]


teacher_registry = {'lookahead': lookahead_discrete_action}


def get_teacher_actions_and_trajectory(
    teacher_type: TeacherType,
    teacher_params: TeacherParams,
    init_pose: Pose4D,
    trajectory: list[Point3D],
) -> tuple[list[DiscreteActionID], list[Pose4D]]:
    
    # decide action type
    teacher = teacher_registry[teacher_type]

    # init states
    actions = []
    trajectory = np.array(trajectory)
    teacher_trajectory = [cur_pose := init_pose]

    # simulate episode with teacher actions
    while (action := teacher(cur_pose, trajectory, teacher_params)) is not DiscreteAction.STOP:
        # accumulate action
        actions.append(action.index)
        teacher_trajectory.append(cur_pose := _moved_pose(cur_pose, *action.value))

        # no looking back
        distances = np.linalg.norm(trajectory - np.array(cur_pose.xyz), axis=-1)
        nearest_waypoint_idx = np.argmin(distances)
        trajectory = trajectory[nearest_waypoint_idx:]
    
    actions.append(DiscreteAction.STOP.index)
    
    return actions, teacher_trajectory


def _moved_pose(cur_pose: Pose4D, stride_forward: float, d_yaw: float, dz:float):
    x, y, z, yaw = cur_pose

    moved_yaw = modulo_radians(yaw + d_yaw)
    dx, dy = stride_forward * np.cos(moved_yaw), stride_forward * np.sin(moved_yaw)
    
    return Pose4D(x + dx, y + dy, z + dz, moved_yaw)