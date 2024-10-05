from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast

from vlnce.actions import DiscreteAction
from vlnce.aux_losses import AuxLosses
from vlnce.dataset.episode import Episode
from vlnce.observation import cropclient
from vlnce.observation.airsimclient import AirsimClient
from vlnce.parser import ExperimentArgs
from vlnce.policy import Policy
from vlnce.space import Point3D, Pose4D, modulo_radians

ACTION_ID = int
EPISODE_ID = tuple[str, int, int]
ACTION_LOGS = dict[EPISODE_ID, list[ACTION_ID]]
TRAJECTORY_LOGS = dict[EPISODE_ID, list[Pose4D]]


@dataclass
class EvaluationMetrics:
    navigation_error: float = np.inf
    success_rate: float = 0.
    oracle_success_rate: float = 0.
    
    @classmethod
    def names(cls):
        return list(asdict(cls()))
    
    def to_dict(self):
        return asdict(self)


def eval_policy(
    policy: Policy,
    episodes: list[Episode],
    args: ExperimentArgs,
    device: str,
    return_logs=False,
):

    action_logs, trajectory_logs = (run_episodes if args.eval_batch_size == 1 else run_episodes_batch)(policy, episodes, args, device)

    navigation_error_by_episode = np.array([trajectory_logs[eps.id][-1].xyz.dist_to(eps.target_position) for eps in episodes])
    navigation_error = navigation_error_by_episode.mean()
    success_rate = (navigation_error_by_episode < args.success_dist).mean()


    def oracle_distance(goal: Point3D, trajectory: list[Pose4D]) -> float:
        goal = np.array(goal.xy)
        trajectory = np.array(trajectory)[:, :2]
        distances = np.linalg.norm(goal - trajectory, axis=-1)
        return distances.min()

    oracle_distance_by_episode = np.array([oracle_distance(eps.target_position, trajectory_logs[eps.id]) for eps in episodes])
    oracle_success_rate = (oracle_distance_by_episode <= args.success_dist).mean()

    metrics = EvaluationMetrics(navigation_error, success_rate, oracle_success_rate)

    if return_logs:
        return action_logs, trajectory_logs, metrics
    else:
        return metrics


@torch.no_grad()
def run_episodes_batch(
    policy: Policy,
    episodes: list[Episode],
    args: ExperimentArgs,
    device: str,
) -> tuple[ACTION_LOGS, TRAJECTORY_LOGS]:
    
    assert args.eval_client != 'airsim', "airsim cannot be run in batch"
    
    if reactivate := AuxLosses.is_active():
        AuxLosses.deactivate()

    # load data
    cropclient.load_image_cache()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    dataloader = DataLoader(episodes, args.eval_batch_size, shuffle=False, collate_fn=lambda x: x)

    action_logs = defaultdict(list)
    trajectory_logs = defaultdict(list)
    
    episodes_batch: list[Episode]
    for episodes_batch in tqdm(dataloader, desc='eval episodes', unit='batch', colour='#88dd88'):
        
        # init episode
        batch_size = len(episodes_batch)
        poses = [eps.start_pose for eps in episodes_batch]
        instructions = tokenizer(
            [eps.target_description for eps in episodes_batch],
            padding=True,
            return_attention_mask=False, return_token_type_ids=False, return_tensors='pt'
        )['input_ids'].int().to(device)
        rnn_states = policy.get_initial_recurrent_hidden_states(batch_size, device)
        actions = torch.full((batch_size,), DiscreteAction.STOP.index, device=device)
        not_dones = torch.full((batch_size,), True).to(device)

        for t in range(args.eval_max_timestep + 1):
            
            # log
            for eps, action, pose, not_done in zip(episodes_batch, actions, poses, not_dones):
                if not_done:
                    action_logs[eps.id].append(action.item())
                    trajectory_logs[eps.id].append(pose)
            
            if t > 0 and (~not_dones).all().item():
                break
            
            # step
            rgbs = np.stack([
                cropclient.crop_image(eps.map_name, pose, args.rgb_size, 'rgb') \
                if not_done else np.zeros((*args.rgb_size, 3), dtype=np.uint8)
                for eps, pose, not_done in zip(episodes_batch, poses, not_dones)
            ])
            depths = np.stack([
                cropclient.crop_image(eps.map_name, pose, args.depth_size, 'depth') \
                if not_done else np.zeros((*args.depth_size, 1), dtype=np.float32)
                for eps, pose, not_done in zip(episodes_batch, poses, not_dones)
            ])
            obs = {
                'rgb': torch.tensor(rgbs).float().to(device),
                'depth': torch.tensor(depths).float().to(device),
                'instruction': instructions
            }
            actions, rnn_states = policy.act(obs, rnn_states, actions, not_dones, deterministic=True)
            actions = actions.flatten()
            not_dones = not_dones & (actions != DiscreteAction.STOP.index)
            poses = [
                _moved_pose(pose, DiscreteAction.from_index(action.item())) if mask else pose
                for pose, action, mask in zip(poses, actions, not_dones)
            ]
            

    if reactivate:
        AuxLosses.activate()
    
    return action_logs, trajectory_logs



@torch.no_grad()
def run_episodes(
    policy: Policy,
    episodes: list[Episode],
    args: ExperimentArgs,
    device: str,
) -> tuple[ACTION_LOGS, TRAJECTORY_LOGS]:
    
    if reactivate := AuxLosses.is_active():
        AuxLosses.deactivate()

    # set image client
    if args.eval_client == 'crop':
        cropclient.load_image_cache()
        client = cropclient
    if args.eval_client == 'airsim':
        client = AirsimClient(args.sim_ip, args.sim_port, 'survey')
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    action_logs = defaultdict(list)
    trajectory_logs = defaultdict(list)

    for episode in tqdm(episodes, desc='eval episodes', unit='episode', colour='#88dd88'):
        
        # init episode
        pose = episode.start_pose
        instruction = tokenizer(episode.target_description, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')['input_ids'].int().to(device)
        rnn_states = policy.get_initial_recurrent_hidden_states(1, device)
        action = torch.tensor(DiscreteAction.STOP.index, device=device)
        mask = torch.tensor(True).to(device)

        for t in range(args.eval_max_timestep + 1):
            
            # log
            action_logs[episode.id].append(action.item())
            trajectory_logs[episode.id].append(pose)

            if t > 0 and action.item() == DiscreteAction.STOP.index:
                break
            
            # step
            rgb, depth = client.get_rgbd(episode.map_name, pose, args.rgb_size, args.depth_size)
            obs = {
                'rgb': torch.tensor(rgb).unsqueeze(0).float().to(device),
                'depth': torch.tensor(depth).unsqueeze(0).float().to(device),
                'instruction': instruction
            }
            action, rnn_states = policy.act(obs, rnn_states, action, mask, deterministic=True)
            pose = _moved_pose(pose, DiscreteAction.from_index(action.item()))
        print(t)
    if reactivate:
        AuxLosses.activate()
    
    return action_logs, trajectory_logs


def _moved_pose(pose: Pose4D, action: DiscreteAction):
    x, y , z, yaw = pose
    d_forward, d_yaw, dz = action.value

    yaw = modulo_radians(yaw + d_yaw)
    dx = d_forward * np.cos(yaw)
    dy = d_forward * np.sin(yaw)

    return Pose4D(x + dx, y + dy, z + dz, yaw)
