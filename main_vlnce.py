import torch

from vlnce.parser import parse_args
from vlnce.train import train
from vlnce.policy import CMAPolicy, Seq2SeqPolicy
from vlnce.evaluate import eval_policy
from vlnce.dataset.generate import generate_episodes_from_mturk_trajectories
from vlnce.dataset.mturk_trajectory import load_mturk_trajectories
from vlnce.cityreferobject import get_city_refer_objects


DEVICE = 'cuda'


args = parse_args()

if args.mode == 'train':

    train(args, DEVICE)

if args.mode == 'eval':

    model_trajectory = args.checkpoint.split('/')[-2]
    epoch = args.checkpoint.split('/')[-1].split('.')[0]

    objects = get_city_refer_objects()

    # load policy
    PolicyType = CMAPolicy if args.policy == 'cma' else Seq2SeqPolicy
    policy = PolicyType(args.rgb_size, args.depth_size, args.use_progress_monitor).to(DEVICE).eval()
    if args.checkpoint:
        policy.load_state_dict(torch.load(args.checkpoint)['policy_state_dict'])

    for split in ('val_seen', 'val_unseen', 'test_unseen'):

        test_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories(split, 'all'))
        action_logs, trajectory_logs, eval_metrics = eval_policy(policy, test_episodes, args, DEVICE, return_logs=True)

        print(f"{split} -- {eval_metrics.navigation_error: .1f}, {eval_metrics.success_rate*100: .2f}, {eval_metrics.oracle_success_rate*100: .2f}, {eval_metrics.success_rate_weighted_by_path_length*100: .2f}")

        import json
        with open(f'logs_{model_trajectory}_{split}_{epoch}.json', 'w') as f:
            json.dump({
                'metrics': eval_metrics.to_dict(),
                'action_logs': {str(eps_id): actions for eps_id, actions in action_logs.items()},
                'trajectory_logs': {str(eps_id): [tuple(pose) for pose in trajectory] for eps_id, trajectory in trajectory_logs.items()}
            }, f)

    