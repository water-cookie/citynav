import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange

from gsamllavanav.defaultpaths import GOAL_PREDICTOR_CHECKPOINT_DIR
from gsamllavanav.cityreferobject import get_city_refer_objects, MultiMapObjects
from gsamllavanav.dataset.episode import Episode
from gsamllavanav.dataset.generate import convert_trajectory_to_shortest_path, generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.mapdata import MAP_BOUNDS
from gsamllavanav.observation import cropclient
from gsamllavanav.models.goal_predictor import GoalPredictor
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from gsamllavanav.space import Point2D
from gsamllavanav.evaluate import eval_goal_predictor, run_episodes_batch, GoalPredictorMetrics
from gsamllavanav import logger


def train(args: ExperimentArgs, device='cuda'):

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # setup logger
    logger.init(args)
    for metric in GoalPredictorMetrics.names():
        logger.define_metric('val_seen_' + metric, 'epoch')
        logger.define_metric('val_unseen_' + metric, 'epoch')

    # load data
    start_epoch = 0
    objects = get_city_refer_objects()
    train_episodes = _load_train_episodes(objects, args)
    if args.train_episode_sample_size > 0:
        train_episodes = random.sample(train_episodes, args.train_episode_sample_size)
    train_dataloader = DataLoader(train_episodes, args.train_batch_size, shuffle=True, collate_fn=lambda x: x)
    val_seen_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('val_seen', 'all', args.altitude))
    val_unseen_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('val_unseen', 'all', args.altitude))
    cropclient.load_image_cache()

    # init model & optim
    goal_predictor = GoalPredictor(args.map_size).to(device)
    optimizer = AdamW(goal_predictor.parameters(), args.learning_rate)
    if args.checkpoint:
        start_epoch, goal_predictor, optimizer = _load_checkpoint(goal_predictor, optimizer, args)

    if args.eval_at_start:
        _eval_predictor_and_log_metrics(goal_predictor, val_seen_episodes, val_unseen_episodes, args, device)

    episodes_batch: list[Episode]
    for epoch in trange(start_epoch, args.epochs, desc='epochs', unit='epoch', colour='#448844'):
        for episodes_batch in tqdm(train_dataloader, desc='train episodes', unit='batch', colour='#88dd88'):
            
            maps, rgbs, normalized_depths = prepare_inputs(episodes_batch, args, device)
            normalized_goal_xys, progresses = prepare_labels(episodes_batch, args, device)
            
            pred_normalized_goal_xys, pred_progresses = goal_predictor(maps, rgbs, normalized_depths, flip_depth=True)
            
            goal_prediction_loss = F.mse_loss(pred_normalized_goal_xys, normalized_goal_xys)
            progress_loss = F.mse_loss(pred_progresses, progresses)
            loss = goal_prediction_loss + progress_loss
            loss.backward()
            logger.log({
                'loss': loss.item(),
                'goal_prediction_loss': goal_prediction_loss.item(),
                'progress_loss': progress_loss.item()
            })

            optimizer.step()
            optimizer.zero_grad()

        logger.log({'epoch': epoch})
        
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(epoch, goal_predictor, optimizer, args)
        
        if (epoch + 1) % args.eval_every == 0:
            _eval_predictor_and_log_metrics(goal_predictor, val_seen_episodes, val_unseen_episodes, args, device)


def normalize_position(pos: Point2D, map_name: str, map_meters: float):
    return (pos.x - MAP_BOUNDS[map_name].x_min) / map_meters, (MAP_BOUNDS[map_name].y_max - pos.y) / map_meters


def prepare_inputs(episodes_batch: list[Episode], args: ExperimentArgs, device: str):

    maps = np.concatenate([
        LandmarkNavMap.generate_maps_for_an_episode(
            episode, args.map_shape, args.map_pixels_per_meter, args.map_update_interval, args.gsam_rgb_shape, args.gsam_params, args.gsam_use_map_cache
        )
        for episode in episodes_batch
    ])

    rgbs = np.stack([
        cropclient.crop_image(episode.map_name, pose, (224, 224), 'rgb')
        for episode in episodes_batch
        for pose in episode.sample_trajectory(args.map_update_interval)
    ]).transpose(0, 3, 1, 2)

    normalized_depths = np.stack([
        cropclient.crop_image(episode.map_name, pose, (256, 256), 'depth')
        for episode in episodes_batch
        for pose in episode.sample_trajectory(args.map_update_interval)
    ]).transpose(0, 3, 1, 2) / args.max_depth

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

    return maps, rgbs, normalized_depths


def prepare_labels(episodes_batch: list[Episode], args: ExperimentArgs, device: str):
    normalized_goal_xys = [
        normalize_position(episode.target_position, episode.map_name, args.map_meters)
        for episode in episodes_batch
        for _ in episode.sample_trajectory(args.map_update_interval)
    ]

    progresses = [
        np.clip(1 - episode.target_position.xy.dist_to(pose.xy) / episode.target_position.xy.dist_to(episode.start_pose.xy), 0, 1)
        for episode in episodes_batch
        for pose in episode.sample_trajectory(args.map_update_interval)
    ]
    
    normalized_goal_xys = torch.tensor(normalized_goal_xys, device=device, dtype=torch.float32)
    progresses = torch.tensor(progresses, device=device, dtype=torch.float32).reshape(-1, 1)

    return normalized_goal_xys, progresses


def _load_train_episodes(objects: MultiMapObjects, args: ExperimentArgs) -> list[Episode]:
    mturk_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('train_seen', 'all', args.altitude))
    if args.train_trajectory_type == 'mturk':
        return mturk_episodes
    if args.train_trajectory_type == 'sp':
        return [convert_trajectory_to_shortest_path(eps, 'linear_xy') for eps in tqdm(mturk_episodes, desc='converting to shortest path episode')]
    if args.train_trajectory_type == 'both':
        return mturk_episodes + [convert_trajectory_to_shortest_path(eps, 'linear_xy') for eps in tqdm(mturk_episodes, desc='converting to shortest path episode')]


def _eval_predictor_and_log_metrics(
    goal_predictor: GoalPredictor,
    val_seen_episodes: list[Episode],
    val_unseen_episodes: list[Episode],
    args: ExperimentArgs,
    device: str,
):
    val_seen_metrics = eval_goal_predictor(args, val_seen_episodes, *run_episodes_batch(args, goal_predictor, val_seen_episodes, device))
    val_unseen_metrics = eval_goal_predictor(args, val_unseen_episodes,  *run_episodes_batch(args, goal_predictor, val_unseen_episodes, device))
    logger.log({'val_seen_' + k: v for k, v in val_seen_metrics.to_dict().items()})
    logger.log({'val_unseen_' + k: v for k, v in val_unseen_metrics.to_dict().items()})


def _load_checkpoint(
    goal_predictor: GoalPredictor,
    optimizer: torch.optim.Optimizer,
    args: ExperimentArgs,
):
    checkpoint = torch.load(args.checkpoint)
    start_epoch: int = checkpoint['epoch'] + 1
    goal_predictor.load_state_dict(checkpoint['predictor_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return start_epoch, goal_predictor, optimizer


def _save_checkpoint(
    epoch: int,
    goal_predictor: GoalPredictor,
    optimizer: torch.optim.Optimizer,
    args: ExperimentArgs,
):
    ablation = f"-{args.ablate}" if args.ablate else ''
    train_size = '' if args.train_episode_sample_size < 0 else f"_{args.train_episode_sample_size}"
    checkpoint_dir = GOAL_PREDICTOR_CHECKPOINT_DIR/f"{args.train_trajectory_type}_{args.altitude}_{args.gsam_box_threshold}{ablation}{train_size}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    torch.save(
        {
            'epoch': epoch,
            'predictor_state_dict': goal_predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        checkpoint_dir/f"{epoch:03d}.pth"
    )