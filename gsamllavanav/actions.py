from enum import Enum
from math import pi
from typing import NamedTuple


class Action(NamedTuple):
    forward_movement_meters: float
    yaw_radians: float
    vertical_movement_meters: float


class DiscreteAction(Enum):
    STOP = Action(0, 0, 0)
    MOVE_FORWARD = Action(5, 0, 0)
    TURN_RIGHT = Action(0,-pi/6, 0)
    TURN_LEFT = Action(0, pi/6, 0)
    GO_UP = Action(0, 0, 2)
    GO_DOWN = Action(0, 0, -2)
    
    @property
    def index(self):
        return {discrete_action: i for i, discrete_action in enumerate(DiscreteAction)}[self]

    @classmethod
    def from_index(cls, idx: int):
        return list(cls)[idx]
