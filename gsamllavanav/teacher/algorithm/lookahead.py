from dataclasses import dataclass

import numpy as np

from gsamllavanav.actions import Action, DiscreteAction
from gsamllavanav.space import Point3D, Pose4D, modulo_radians


@dataclass
class LookaheadTeacherParams:
    '''Parameters for `lookahead_discrete_action()`
    
    Attributes
    ----------
    lookahead : int
        the number of points to look ahead from the current nearest point of the trajectory
    stop_distance_meters : float
        stops if the goal is within the distance of this value
    turn_threshold_radians : float
        turns left/right if the next point deviates more than this angle from the front direction
    vertical_movement_threshold_meters : float
        moves up/down if the next point deviates more than this value in z direction
    '''
    lookahead: int = 3
    stop_distance_meters: float = 5.
    turn_threshold_radians: float = np.pi/6
    vertical_movement_threshold_meters: float = 1.


def lookahead_discrete_action(
    current_pose: Pose4D,
    trajectory: list[Point3D],
    params: LookaheadTeacherParams = LookaheadTeacherParams()
):
    '''returns `DiscreteAction` to move towards the next point in `trajectory`'''
    
    current_position = np.array(current_pose.xyz)
    trajectory = np.array(trajectory)

    distance_to_goal = np.linalg.norm(trajectory[-1] - current_position)

    if distance_to_goal < params.stop_distance_meters:
        return DiscreteAction.STOP
    
    next_action = lookahead_continuous_action(current_pose, trajectory, params.lookahead)

    if next_action.vertical_movement_meters > params.vertical_movement_threshold_meters:
        return DiscreteAction.GO_UP
    if next_action.vertical_movement_meters < -params.vertical_movement_threshold_meters:
        return DiscreteAction.GO_DOWN

    if next_action.yaw_radians > params.turn_threshold_radians:
        return DiscreteAction.TURN_LEFT
    if next_action.yaw_radians < -params.turn_threshold_radians:
        return DiscreteAction.TURN_RIGHT
    
    return DiscreteAction.MOVE_FORWARD


def lookahead_continuous_action(
    current_pose: Pose4D,
    trajectory: list[Point3D],
    lookahead: int,
):
    '''returns `Action` required to reach the next point in `trajectory`'''

    current_position = np.array(current_pose.xyz)
    trajectory = np.array(trajectory)

    # get the next waypoint
    distance_to_waypoints = np.linalg.norm(trajectory - current_position, axis=-1)
    closest_waypoint_idx = np.argmin(distance_to_waypoints)
    next_waypoint_idx = np.clip(closest_waypoint_idx + lookahead, 0, len(trajectory) - 1)
    next_waypoint = trajectory[next_waypoint_idx]

    # forward movement
    dx, dy, dz = next_waypoint - current_position
    forward_stride = np.linalg.norm((dx, dy))

    # yaw angle
    next_yaw = np.arctan2(dy, dx)
    d_yaw = 0 if forward_stride < 0.01 else modulo_radians(next_yaw - current_pose.yaw)

    return Action(forward_stride, d_yaw, dz)
