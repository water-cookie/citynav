import copy
import json
from dataclasses import dataclass
from typing import Literal, Optional

from gsamllavanav.space import Point3D, Pose5D
from gsamllavanav.defaultpaths import MTURK_TRAJECTORY_DIR
from gsamllavanav.trajectory import straight_line_trajectory
from gsamllavanav.mapdata import GROUND_LEVEL


MturkSplit = Literal['train_seen', 'val_seen', 'val_unseen', 'test_unseen']
MturkDifficulty = Literal['easy', 'medium', 'hard', 'all']


def load_mturk_trajectories(split: MturkSplit, difficulty: MturkDifficulty, fix_altitude: Optional[float] = None, trajectory_dir=MTURK_TRAJECTORY_DIR):
    
    difficulty = '' if difficulty == 'all' else  '_' + difficulty
    path = trajectory_dir/f"citynav_{split}{difficulty}.json"
    
    with open(path) as f:
        trajectories =  [MTurkTrajectory(**trajectory) for trajectory in json.load(f)]
        
    if fix_altitude:
        trajectories =  [t.fix_altitude(fix_altitude) for t in trajectories]
    
    return trajectories



@dataclass
class MTurkTrajectory:
    area: str
    block: str
    object_ids: list[int]
    ann_ids: list[int]
    descriptions: list[str]
    trajectory: list[Pose5D]
    marker_positions: list[Point3D]
    target_positions: list[Point3D]
    total_score: float
    dist_marker_to_target: float
    split: str
    dist_start_to_target: float = None

    def __post_init__(self):
        if len(self.trajectory[0]) == 5:
            self.trajectory = [Pose5D(x, y, z, yaw, pitch) for x, y, z, yaw, pitch in self.trajectory]
        if len(self.trajectory[0]) == 6:
            self.trajectory = [Pose5D.from_direction_vector(x, y, z, dx, dy, dz) for x, y, z, dx, dy, dz in self.trajectory]
        
        self.marker_positions = [Point3D(x, y, z) for x, y, z in self.marker_positions]
        self.target_positions = [Point3D(x, y, z) for x, y, z in self.target_positions]

        if self.dist_start_to_target is None:
            self.dist_start_to_target = self.start_pose.xyz.dist_to(self.target_position)
    
    @property
    def map_name(self):
        return f"{self.area}_block_{self.block}"

    @property
    def object_id(self):
        return self.object_ids[0]
    
    @property
    def start_pose(self):
        return self.trajectory[0].xyzyaw
    
    @property
    def target_position(self):
        return self.target_positions[-1]

    @property
    def desc_id(self):
        return self.ann_ids[0]
    
    @property
    def trajectory_xyz(self):
        return [pose.xyz for pose in self.trajectory]

    @property
    def extended_trajectory(self):
        return self.trajectory_xyz + [self.marker_positions[-1]]
    
    @property
    def interpolated_trajectory(self):
        interploated_trajectory = [self.start_pose.xyz]
        for src, dst in zip(self.extended_trajectory[:-1], self.extended_trajectory[1:]):
            for pos in straight_line_trajectory(src, dst):
                if interploated_trajectory[-1].dist_to(pos) > 5.:
                    interploated_trajectory.append(pos)
        
        return interploated_trajectory
    
    def fix_altitude(self, altitude_from_ground: float):
        
        new_trajectory = copy.deepcopy(self)

        ground_level = GROUND_LEVEL[self.map_name]
        new_z = ground_level + altitude_from_ground
        
        new_trajectory.trajectory = [Pose5D(x, y, new_z, yaw, pitch) for x, y, z, yaw, pitch in self.trajectory]
        new_trajectory.marker_positions = [Point3D(x, y , new_z) for x, y, z in self.marker_positions]
        new_trajectory.dist_start_to_target = new_trajectory.start_pose.xy.dist_to(self.target_position.xy)

        return new_trajectory
