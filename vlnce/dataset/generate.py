from tqdm import tqdm

from vlnce.cityreferobject import MultiMapObjects
from vlnce.dataset.episode import Episode
from vlnce.dataset.mturk_trajectory import MTurkTrajectory
from vlnce.teacher.algorithm.lookahead import LookaheadTeacherParams
from vlnce.teacher.trajectory import TeacherParams, TeacherType, get_teacher_actions_and_trajectory
from vlnce.trajectory import TrajectoryType, trajectory_registry


def generate_episodes_from_mturk_trajectories(
    objects: MultiMapObjects,
    mturk_trajectories: list[MTurkTrajectory],
    max_dist_marker_to_target=30,
    max_steps=500,
    teacher_type: TeacherType = 'lookahead',
    teacher_params: TeacherParams = LookaheadTeacherParams(lookahead=1),
) -> list[Episode]:
    episodes = []
    for mturk_traj in tqdm(mturk_trajectories, desc='generating episodes'):

        if mturk_traj.dist_marker_to_target > max_dist_marker_to_target:
            continue

        teacher_actions, teacher_trajectory = get_teacher_actions_and_trajectory(
            teacher_type, teacher_params, mturk_traj.start_pose, mturk_traj.interpolated_trajectory
        )

        if len(teacher_actions) <= max_steps:
            episodes.append(Episode(objects[mturk_traj.map_name][mturk_traj.object_id], mturk_traj.desc_id, teacher_trajectory, teacher_actions))
    
    return episodes


def convert_trajectory_to_shortest_path(
    episode: Episode,
    trajectory_type: TrajectoryType = 'move_and_drop',
    use_teacher_dst=False,
    teacher_type: TeacherType = 'lookahead',
    teacher_params: TeacherParams = LookaheadTeacherParams(),
) -> Episode:
    dst = episode.teacher_trajectory[-1].xyz if use_teacher_dst else episode.target_position
    sp_trajectory = trajectory_registry[trajectory_type](episode.start_pose.xyz, dst)
    teacher_actions, teacher_trajectory = get_teacher_actions_and_trajectory(
        teacher_type, teacher_params, episode.start_pose, sp_trajectory
    )

    return Episode(
        episode.target_object,
        episode.description_id,
        teacher_trajectory,
        teacher_actions,
    )