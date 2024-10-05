import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
from transformers import BertTokenizerFast

from vlnce import logger
from vlnce.aux_losses import AuxLosses
from vlnce.cityreferobject import get_city_refer_objects, MultiMapObjects
from vlnce.dataset.episode import Episode
from vlnce.dataset.generate import convert_trajectory_to_shortest_path, generate_episodes_from_mturk_trajectories
from vlnce.dataset.mturk_trajectory import load_mturk_trajectories
from vlnce.defaultpaths import CHECKPOINT_DIR
from vlnce.evaluate import eval_policy, EvaluationMetrics
from vlnce.observation import cropclient
from vlnce.parser import ExperimentArgs
from vlnce.policy import CMAPolicy, Seq2SeqPolicy, Policy


def train(args: ExperimentArgs, device='cuda'):

    torch.manual_seed(0)

    # setup logger
    logger.init(args)
    for metric in EvaluationMetrics.names():
        logger.define_metric('val_seen_' + metric, 'epoch')
        logger.define_metric('val_unseen_' + metric, 'epoch')
    
    # load data
    start_epoch = 0
    objects = get_city_refer_objects()
    train_dataloader = DataLoader(_load_train_episodes(objects, args), args.train_batch_size, shuffle=True, collate_fn=lambda x: x)
    val_seen_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('val_seen', 'all'))
    val_unseen_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('val_unseen', 'all'))
    cropclient.load_image_cache()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # init/reload policy & optim
    PolicyType = CMAPolicy if args.policy == 'cma' else Seq2SeqPolicy
    policy = PolicyType(args.rgb_size, args.depth_size, args.use_progress_monitor).to(device)
    optimizer = Adam(policy.parameters(), args.learning_rate)
    if args.checkpoint:
        start_epoch, policy, optimizer = _load_checkpoint(policy, optimizer, args)
    
    if args.eval_at_start:
        _eval_policy_and_log_metrics(policy, val_seen_episodes, val_unseen_episodes, args, device)

    if args.use_progress_monitor:
        AuxLosses.activate()

    # training loop
    episodes_batch: list[Episode]
    for epoch in trange(start_epoch, args.epochs, desc='epochs', unit='epoch', colour='#448844'):
        for episodes_batch in tqdm(train_dataloader, desc='train episodes', unit='batch', colour='#88dd88'):
            
            max_timestep = max([eps.time_step for eps in episodes_batch])
            batch_size = len(episodes_batch)
            
            # get inputs
            obs_batch = _get_obs_batch(episodes_batch, args.rgb_size, args.depth_size, tokenizer, device)
            info_batch = _get_info_batch(episodes_batch, device)
            rnn_states = policy.get_initial_recurrent_hidden_states(batch_size, device)

            # compute logits
            AuxLosses.clear()
            distribution = policy.build_distribution(obs_batch, rnn_states, info_batch['previous_actions'], info_batch['not_done_masks'])
            logits = distribution.logits.view(max_timestep, batch_size, -1)  # (T, B, N_ACTIONS)

            # compute loss
            loss, action_loss, aux_loss = _compute_loss(logits, info_batch)
            loss.backward()
            logger.log({
                'loss': loss.item(),
                'action_loss': action_loss.item(),
                'aux_loss': aux_loss.item()
            })

            # step optimizer
            optimizer.step()
            optimizer.zero_grad()

        logger.log({'epoch': epoch})

        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(epoch, policy, optimizer, args)
        
        if (epoch + 1) % args.eval_every == 0:
            _eval_policy_and_log_metrics(policy, val_seen_episodes, val_unseen_episodes, args, device)


def _get_obs_batch(
    episodes_batch: list[Episode],
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int],
    tokenizer: BertTokenizerFast,
    device: str
):
    
    max_timestep = max([eps.time_step for eps in episodes_batch])

    rgb = np.array([
        [
            cropclient.crop_image(eps.map_name, eps.trajectory[t], rgb_size, 'rgb') \
            if t < eps.time_step else \
            np.zeros((*rgb_size, 3), dtype=np.uint8)
            for eps in episodes_batch
        ]
        for t in range(max_timestep)
    ])
    depth = np.array([
        [
            cropclient.crop_image(eps.map_name, eps.trajectory[t], depth_size, 'depth') \
            if t < eps.time_step else \
            np.zeros((*depth_size, 1), dtype=np.uint8)
            for eps in episodes_batch
        ]
        for t in range(max_timestep)
    ])

    instruction: torch.Tensor = tokenizer(
        [episode.target_description for episode in episodes_batch],
        padding=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_tensors='pt',
    )['input_ids']
    instruction = instruction.tile((max_timestep, 1, 1))

    progress = [
        [min(t/eps.time_step, 1) for eps in episodes_batch]
        for t in range(max_timestep)
    ]

    return {
        'rgb': torch.tensor(rgb).flatten(0, 1).float().to(device),
        'depth': torch.tensor(depth).flatten(0, 1).float().to(device),
        'instruction': instruction.flatten(0, 1).int().to(device),
        'progress': torch.tensor(progress).flatten(0, 1).float().to(device),
    }


def _get_info_batch(episodes_batch: list[Episode], device: str):
    
    max_timestep = max([eps.time_step for eps in episodes_batch])
    batch_size = len(episodes_batch)

    teacher_actions = np.array([
        [
            eps.teacher_actions[t] if t < eps.time_step else 0
            for eps in episodes_batch
        ]
        for t in range(max_timestep)
    ])
    previous_actions = np.concatenate((np.zeros((1, batch_size)), teacher_actions[:-1]))
    inflection_weights = np.where(teacher_actions != previous_actions, 6., 1)
    not_done_masks = np.tile(np.arange(max_timestep), (batch_size, 1)).T < np.array([eps.time_step for eps in episodes_batch])

    return {
        'teacher_actions': torch.tensor(teacher_actions).to(device),
        'inflection_weights': torch.tensor(inflection_weights).to(device),
        'previous_actions': torch.tensor(previous_actions).to(device),
        'not_done_masks': torch.tensor(not_done_masks).contiguous().to(device),
    }


def _compute_loss(logits, info_batch):

    action_loss = F.cross_entropy(logits.permute(0, 2, 1), info_batch['teacher_actions'], reduction="none")  # (T, B)
    action_loss *= info_batch['inflection_weights']                                                          # (T, B)
    action_loss = action_loss.sum(0) / info_batch['inflection_weights'].sum(0)                               # (B, )
    action_loss = action_loss.mean()                                                                         # (1, )
    
    aux_loss = AuxLosses.reduce(info_batch['not_done_masks'].view(-1)) if AuxLosses.is_active() else torch.tensor(0, dtype=torch.float32, device=action_loss.device)

    loss = action_loss + aux_loss

    return loss, action_loss, aux_loss


def _load_train_episodes(objects: MultiMapObjects, args: ExperimentArgs) -> list[Episode]:
    
    mturk_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('train_seen', 'all'))

    if args.train_trajectory_type == 'mturk':
        return mturk_episodes
    if args.train_trajectory_type == 'sp':
        return [convert_trajectory_to_shortest_path(eps) for eps in tqdm(mturk_episodes, desc='converting to shortest path episode')]
    if args.train_trajectory_type == 'both':
        return mturk_episodes + [convert_trajectory_to_shortest_path(eps) for eps in tqdm(mturk_episodes, desc='converting to shortest path episode')]


def _eval_policy_and_log_metrics(
    policy: Policy,
    val_seen_episodes: list[Episode],
    val_unseen_episodes: list[Episode],
    args: ExperimentArgs,
    device: str,
):
    val_seen_metrics = eval_policy(policy, val_seen_episodes, args, device)
    val_unseen_metrics = eval_policy(policy, val_unseen_episodes, args, device)
    logger.log({'val_seen_' + k: v for k, v in val_seen_metrics.to_dict().items()})
    logger.log({'val_unseen_' + k: v for k, v in val_unseen_metrics.to_dict().items()})


def _load_checkpoint(
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    args: ExperimentArgs,
):
    checkpoint = torch.load(args.checkpoint)
    start_epoch: int = checkpoint['epoch'] + 1
    policy.load_state_dict(checkpoint['policy_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return start_epoch, policy, optimizer


def _save_checkpoint(
    epoch: int,
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    args: ExperimentArgs,
):
    checkpoint_dir = CHECKPOINT_DIR/f"{args.policy}_{args.train_trajectory_type}"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'policy_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        checkpoint_dir/f"{epoch:03d}.pth"
    )