import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
from transformers import BertTokenizerFast

from gsamllavanav.defaultpaths import BASELINE_WITH_MAP_CHECKPOINT_DIR
from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.episode import Episode
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.observation import cropclient
from gsamllavanav.models.cma_with_map import CMAwithMap
from gsamllavanav.models.seq2seq_with_map import Seq2SeqwithMap
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from gsamllavanav.evaluate import eval_goal_predictor, GoalPredictorMetrics
from gsamllavanav.evaluate_baseline_with_map import run_episodes_batch
from gsamllavanav import logger
from gsamllavanav.train import prepare_labels, _load_train_episodes


BaselineModelwithMap = {
    'cma_with_map': CMAwithMap,
    'seq2seq_with_map': Seq2SeqwithMap,
}


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
    baseline_model_with_map : Seq2SeqwithMap | CMAwithMap = BaselineModelwithMap[args.model](args.map_size).to(device)
    optimizer = AdamW(baseline_model_with_map.parameters(), args.learning_rate)
    if args.checkpoint:
        start_epoch, baseline_model_with_map, optimizer = _load_checkpoint(baseline_model_with_map, optimizer, args)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    if args.eval_at_start:
        _eval_predictor_and_log_metrics(baseline_model_with_map, val_seen_episodes, val_unseen_episodes, args, device)

    episodes_batch: list[Episode]
    for epoch in trange(start_epoch, args.epochs, desc='epochs', unit='epoch', colour='#448844'):
        for episodes_batch in tqdm(train_dataloader, desc='train episodes', unit='batch', colour='#88dd88'):
            
            maps, rgbs, normalized_depths, instructions = prepare_inputs(episodes_batch, tokenizer, args, device)
            normalized_goal_xys, progresses = prepare_labels(episodes_batch, args, device)
            rnn_states = baseline_model_with_map.get_initial_recurrent_hidden_states(maps.shape[0], device)
            not_done_masks = torch.ones(maps.shape[0], dtype=bool, device=device)

            pred_normalized_goal_xys, pred_progresses, rnn_states = baseline_model_with_map(instructions, normalized_depths, rgbs, maps, rnn_states, not_done_masks)
                
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
            _save_checkpoint(epoch, baseline_model_with_map, optimizer, args)
        
        if (epoch + 1) % args.eval_every == 0:
            _eval_predictor_and_log_metrics(baseline_model_with_map, val_seen_episodes, val_unseen_episodes, args, device)


def prepare_inputs(episodes_batch: list[Episode], tokenizer: BertTokenizerFast, args: ExperimentArgs, device: str):

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

    instructions : torch.Tensor = tokenizer(
        [
            episode.target_description 
            for episode in episodes_batch
            for _ in episode.sample_trajectory(args.map_update_interval)
        ],
        padding=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_tensors='pt',
    )['input_ids']


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
    instructions = instructions.to(device)

    return maps, rgbs, normalized_depths, instructions


def _eval_predictor_and_log_metrics(
    baseline_model_with_map: CMAwithMap | Seq2SeqwithMap,
    val_seen_episodes: list[Episode],
    val_unseen_episodes: list[Episode],
    args: ExperimentArgs,
    device: str,
):
    val_seen_metrics = eval_goal_predictor(args, val_seen_episodes, *run_episodes_batch(args, baseline_model_with_map, val_seen_episodes, device))
    val_unseen_metrics = eval_goal_predictor(args, val_unseen_episodes,  *run_episodes_batch(args, baseline_model_with_map, val_unseen_episodes, device))
    logger.log({'val_seen_' + k: v for k, v in val_seen_metrics.to_dict().items()})
    logger.log({'val_unseen_' + k: v for k, v in val_unseen_metrics.to_dict().items()})


def _load_checkpoint(
    baseline_model_with_map: CMAwithMap | Seq2SeqwithMap,
    optimizer: torch.optim.Optimizer,
    args: ExperimentArgs,
):
    checkpoint = torch.load(args.checkpoint)
    start_epoch: int = checkpoint['epoch'] + 1
    baseline_model_with_map.load_state_dict(checkpoint['predictor_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return start_epoch, baseline_model_with_map, optimizer


def _save_checkpoint(
    epoch: int,
    baseline_model_with_map: CMAwithMap | Seq2SeqwithMap,
    optimizer: torch.optim.Optimizer,
    args: ExperimentArgs,
):
    ablation = f"-{args.ablate}" if args.ablate else ''
    train_size = '' if args.train_episode_sample_size < 0 else f"_{args.train_episode_sample_size}"
    checkpoint_dir = BASELINE_WITH_MAP_CHECKPOINT_DIR/args.model/f"{args.train_trajectory_type}_{args.altitude}_{args.gsam_box_threshold}{ablation}{train_size}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    torch.save(
        {
            'epoch': epoch,
            'predictor_state_dict': baseline_model_with_map.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        checkpoint_dir/f"{epoch:03d}.pth"
    )