import argparse
from typing import Literal, Optional
from dataclasses import dataclass, asdict


@dataclass
class ExperimentArgs:
    mode: Literal['train', 'eval']

    # logger
    log: bool
    silent: bool
    resume_log_id: str

    # observation
    rgb_size: tuple[int, int]
    depth_size: tuple[int, int]

    # training params
    learning_rate: float
    train_batch_size: int
    epochs: int
    policy: Literal['cma', 'seq2seq']
    use_progress_monitor: bool
    checkpoint: Optional[str]
    save_every: int
    train_trajectory_type: Literal['sp', 'mturk', 'both']
    
    # eval params
    eval_every: int
    eval_batch_size: int
    eval_at_start: bool
    eval_max_timestep: int
    eval_client: Literal['crop', 'airsim']
    success_dist: float
    success_iou: float

    # airsim
    sim_ip: str
    sim_port: int

    def to_dict(self):
        return asdict(self)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')

    # logger
    parser.add_argument('--log', action='store_true', default=False, help="log results to wandb")
    parser.add_argument('--silent', action='store_true', default=False, help="disable printing log info to stdout")
    parser.add_argument('--resume_log_id', type=str, default='')

    # observation
    parser.add_argument('--rgb_size', type=int, default=224)
    parser.add_argument('--depth_size', type=int, default=256)
    
    # training params
    parser.add_argument('--learning_rate', type=float, default=1.0e-03)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--policy', type=str, choices=['cma', 'seq2seq'], default='cma')
    parser.add_argument('--use_progress_monitor', action='store_true', default=False)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--train_trajectory_type', type=str, choices=['sp', 'mturk', 'both'], default='sp')
    
    # eval params
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--eval_at_start', action='store_true', default=False)
    parser.add_argument('--eval_max_timestep', type=int, default=200)
    parser.add_argument('--eval_client', type=str, choices=['crop', 'airsim'], default='crop')
    parser.add_argument('--success_dist', type=float, default=20.)
    parser.add_argument('--success_iou', type=float, default=0.4)

    # airsim
    parser.add_argument('--sim_ip', type=str, default="172.23.96.1")
    parser.add_argument('--sim_port', type=int, default=41451)

    args = parser.parse_args()
    args.rgb_size = args.rgb_size, args.rgb_size
    args.depth_size = args.depth_size, args.depth_size

    return ExperimentArgs(**vars(args))