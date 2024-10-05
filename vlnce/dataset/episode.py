from dataclasses import dataclass

from vlnce.cityreferobject import CityReferObject
from vlnce.space import Pose4D


@dataclass
class Episode:
    target_object: CityReferObject
    description_id: int
    teacher_trajectory: list[Pose4D]
    teacher_actions: list[int]

    @property
    def id(self):
        return self.map_name, self.target_object.id, self.description_id

    @property
    def map_name(self):
        return self.target_object.map_name
    
    @property
    def start_pose(self):
        return self.teacher_trajectory[0]
    
    @property
    def target_position(self):
        return self.target_object.position
    
    @property
    def target_description(self):
        return self.target_object.descriptions[self.description_id]

    @property
    def time_step(self):
        return len(self.teacher_actions)
    
    @property
    def trajectory(self):
        return self.teacher_trajectory
    