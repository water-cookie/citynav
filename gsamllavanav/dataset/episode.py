from dataclasses import dataclass

from gsamllavanav.cityreferobject import CityReferObject
from gsamllavanav.space import Pose4D


MapName = str
ObjectID = int
DescriptionID = int
EpisodeID = tuple[MapName, ObjectID, DescriptionID]


@dataclass
class Episode:
    target_object: CityReferObject
    description_id: int
    teacher_trajectory: list[Pose4D]
    teacher_actions: list[int]

    @property
    def description_landmarks(self):
        return self.target_object.processed_descriptions[self.description_id].landmarks
    
    @property
    def description_surroundings(self):
        return self.target_object.processed_descriptions[self.description_id].surroundings

    @property
    def description_target(self):
        return self.target_object.processed_descriptions[self.description_id].target

    @property
    def id(self) -> EpisodeID:
        return self.map_name, self.target_object.id, self.description_id

    @property
    def map_name(self):
        return self.target_object.map_name
    
    @property
    def start_pose(self):
        return self.teacher_trajectory[0]
    
    @property
    def target_description(self):
        return self.target_object.descriptions[self.description_id]

    @property
    def target_position(self):
        return self.target_object.position
    
    @property
    def target_processed_description(self):
        return self.target_object.processed_descriptions[self.description_id]

    @property
    def target_type(self):
        return self.target_object.object_type

    @property
    def time_step(self):
        return len(self.teacher_actions)
    
    @property
    def trajectory(self):
        return self.teacher_trajectory
    
    def sample_trajectory(self, interval: int):
        return self.teacher_trajectory[::interval] + [self.teacher_trajectory[-1]]
    