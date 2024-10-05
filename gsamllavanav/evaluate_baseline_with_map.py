from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
from transformers import BertTokenizerFast

from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.dataset.episode import Episode, EpisodeID
from gsamllavanav.observation import cropclient
from gsamllavanav.models.cma_with_map import CMAwithMap
from gsamllavanav.models.seq2seq_with_map import Seq2SeqwithMap
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from gsamllavanav.space import Point2D, Pose4D
from gsamllavanav.evaluate import move, unnormalize_position


@torch.no_grad()
def run_episodes_batch(
    args: ExperimentArgs,
    baseline_model_with_map: CMAwithMap | Seq2SeqwithMap,
    episodes: list[Episode],
    device: str,
):
    cropclient.load_image_cache(alt_env=args.alt_env)
    dataloader = DataLoader(episodes, args.eval_batch_size, shuffle=False, collate_fn=lambda x: x)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
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
        instructions : torch.Tensor = tokenizer(
            [episode.target_description for episode in episodes_batch],
            padding=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors='pt',
        )['input_ids'].to(device)
        rnn_states = baseline_model_with_map.get_initial_recurrent_hidden_states(batch_size, device)

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
            not_dones = torch.from_numpy(~dones).to(device=device)

            # predict
            pred_normalized_goal_xys, pred_progresses, rnn_states = baseline_model_with_map(instructions, normalized_depths, rgbs, maps, rnn_states, not_dones)
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
