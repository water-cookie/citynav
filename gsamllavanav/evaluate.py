from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange

from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.dataset.episode import Episode, EpisodeID
from gsamllavanav.observation import cropclient
from gsamllavanav.models.goal_predictor import GoalPredictor
from gsamllavanav.mapdata import MAP_BOUNDS
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from gsamllavanav.space import Point2D, Point3D, Pose4D
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from gsamllavanav.teacher.trajectory import _moved_pose


@dataclass
class GoalPredictorMetrics:
    mean_final_pos_to_goal_dist: float = np.inf
    mean_final_pred_to_goal_dist: float = np.inf
    success_rate_final_pos_to_goal: float = 0.
    success_rate_final_pred_to_goal: float = 0.
    mean_oracle_pos_to_goal_dist: float = np.inf
    mean_oracle_pred_to_goal_dist: float = np.inf
    success_rate_oracle_pos_to_goal: float = 0.
    success_rate_oracle_pred_to_goal: float = 0.
    mean_progress_mse: float = np.inf
    mean_final_progress_mse: float = np.inf
    
    @classmethod
    def names(cls):
        return list(asdict(cls()))
    
    def to_dict(self):
        return asdict(self)


def eval_goal_predictor(
    args: ExperimentArgs,
    episodes: list[Episode],
    trajectory_logs: dict[EpisodeID, list[Pose4D]],
    pred_goal_logs: dict[EpisodeID, list[Point2D]],
    pred_progress_logs: dict[EpisodeID, list[float]],
):
    # metrics based on distance to goal
    final_pos_to_goal_dists = np.array([trajectory_logs[eps.id][-1].xy.dist_to(eps.target_position.xy) for eps in episodes])
    final_pred_to_goal_dists = np.array([pred_goal_logs[eps.id][-1].dist_to(eps.target_position.xy) for eps in episodes])

    # metrics based on path
    def oracle_distance(goal: Point2D, trajectory: list[Point2D]) -> float:
        goal = np.array(goal)
        trajectory = np.array(trajectory)
        distances = np.linalg.norm(goal - trajectory, axis=-1)
        return distances.min()
    
    oracle_pos_to_goal_dists = np.array([oracle_distance(eps.target_position.xy, [pose.xy for pose in trajectory_logs[eps.id]]) for eps in episodes])
    oracle_pred_to_goal_dists = np.array([oracle_distance(eps.target_position.xy, pred_goal_logs[eps.id]) for eps in episodes])

    mean_progress_mse = np.mean([
        np.mean([
            ((1 - min(eps.target_position.xy.dist_to(pose.xy) / eps.target_position.xy.dist_to(eps.start_pose.xy), 1)) - pred_progress) ** 2
            for pose, pred_progress in zip(trajectory_logs[eps.id], pred_progress_logs[eps.id])
        ]) for eps in episodes
    ])
    
    mean_final_progress_mse = np.mean([
        ((1 - min(eps.target_position.xy.dist_to(trajectory_logs[eps.id][-1].xy) / eps.target_position.xy.dist_to(eps.start_pose.xy), 1)) - pred_progress_logs[eps.id][-1]) ** 2
        for eps in episodes
    ])

    metrics = GoalPredictorMetrics(
        final_pos_to_goal_dists.mean(),
        final_pred_to_goal_dists.mean(),
        (final_pos_to_goal_dists <= args.success_dist).mean(),
        (final_pred_to_goal_dists <= args.success_dist).mean(),
        oracle_pos_to_goal_dists.mean(),
        oracle_pred_to_goal_dists.mean(),
        (oracle_pos_to_goal_dists <= args.success_dist).mean(),
        (oracle_pred_to_goal_dists <= args.success_dist).mean(),
        mean_progress_mse, mean_final_progress_mse
    )

    return metrics


@torch.no_grad()
def run_episodes_batch(
    args: ExperimentArgs,
    predictor: GoalPredictor,
    episodes: list[Episode],
    device: str,
):
    cropclient.load_image_cache(alt_env=args.alt_env)
    dataloader = DataLoader(episodes, args.eval_batch_size, shuffle=False, collate_fn=lambda x: x)
    
    pose_logs: dict[EpisodeID, list[Pose4D]] = defaultdict(list)
    pred_goal_logs: dict[EpisodeID, list[Point2D]] = defaultdict(list)
    pred_progress_logs: dict[EpisodeID, list[float]] = defaultdict(list)

    episodes_batch: list[Episode]
    for episodes_batch in tqdm(dataloader, desc='eval episodes', unit='batch', colour='#88dd88', position=1):

        # init episode
        batch_size = len(episodes_batch)
        poses = [eps.start_pose for eps in episodes_batch]
        dones = np.zeros(batch_size, dtype=bool)
        nav_maps = [
            LandmarkNavMap(
                eps.map_name, args.map_shape, args.map_pixels_per_meter,
                eps.description_landmarks, eps.description_target, eps.description_surroundings, args.gsam_params
            ) for eps in episodes_batch
        ]

        for t in trange(args.eval_max_timestep, desc='eval timestep', unit='step', colour='#66aa66', position=2, leave=False):

            gps_noise_batch = np.random.normal(scale=args.gps_noise_scale, size=(batch_size, 2))
            noisy_poses = [Pose4D(x + n_x, y + n_y, z, yaw) for (x, y, z, yaw), (n_x, n_y) in zip(poses, gps_noise_batch)]
            
            # update map
            for eps, pose, noisy_pose, nav_map, done in tqdm(zip(episodes_batch, poses, noisy_poses, nav_maps, dones), desc='updating maps', unit='map', colour='#448844', position=3, leave=False):
                if not done:
                    gsam_rgb = cropclient.crop_image(eps.map_name, pose, args.gsam_rgb_shape, 'rgb')
                    nav_map.update_observations(noisy_pose, gsam_rgb, None, args.gsam_use_map_cache)
                    pose_logs[eps.id].append(pose)

            # prepare inputs
            maps = np.stack([nav_map.to_array() for nav_map in nav_maps])
            rgbs = np.stack([cropclient.crop_image(eps.map_name, pose, (224, 224), 'rgb') for pose in poses]).transpose(0, 3, 1, 2)
            normalized_depths = np.stack([cropclient.crop_image(eps.map_name, pose, (256, 256), 'depth') for pose in poses]).transpose(0, 3, 1, 2) / args.max_depth

            if args.ablate == 'rgb':
                rgbs = np.zeros_like(rgbs)
            if args.ablate == 'depth':
                normalized_depths = np.zeros_like(normalized_depths)
            if args.ablate == 'tracking':
                maps[:, :2] = 0
            if args.ablate == 'landmark':
                maps[:, 2] = 0
            if args.ablate == 'gsam':
                maps[:, 3:] = 0

            maps = torch.tensor(maps, device=device)
            rgbs = torch.tensor(rgbs, device=device)
            normalized_depths = torch.tensor(normalized_depths, device=device, dtype=torch.float32)

            # predict
            pred_normalized_goal_xys, pred_progresses = predictor(maps, rgbs, normalized_depths, flip_depth=True)
            pred_goal_xys = [unnormalize_position(xy.tolist(), eps.map_name, args.map_meters) for eps, xy in zip(episodes_batch, pred_normalized_goal_xys)]
            for eps, done, xy, progress in zip(episodes_batch, dones, pred_goal_xys, pred_progresses.flatten().tolist()):
                if not done:
                    pred_goal_logs[eps.id].append(xy)
                    pred_progress_logs[eps.id].append(progress)

            dones = dones | (pred_progresses.cpu().numpy().flatten() >= args.progress_stop_val)
            
            if dones.all():
                break

            # move
            poses = [
                move(pose, xy, args.move_iteration, noisy_pose) if not done else pose
                for pose, noisy_pose, xy, done in zip(poses, noisy_poses, pred_goal_xys, dones)
            ]

    return dict(pose_logs), dict(pred_goal_logs), dict(pred_progress_logs)


def move(pose: Pose4D, dst: Point2D, iterations: int, noisy_pose: Pose4D):

    dst = Point3D(dst.x, dst.y, pose.z)

    for _ in range(iterations):
        action = lookahead_discrete_action(noisy_pose, [dst])
        pose = _moved_pose(pose, *action.value)
    
    return pose


def unnormalize_position(normalized_xy: tuple[float, float], map_name: str, map_meters: float):
    nx, ny = normalized_xy
    return Point2D(nx * map_meters + MAP_BOUNDS[map_name].x_min, MAP_BOUNDS[map_name].y_max - ny * map_meters)