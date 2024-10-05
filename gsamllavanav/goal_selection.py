from tqdm import tqdm

from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.observation import cropclient
from gsamllavanav.dataset.episode import EpisodeID
from gsamllavanav.mapdata import GROUND_LEVEL
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.space import Point2D, Point3D, Pose4D, bbox_corners_to_position, crwh_to_global_bbox, view_area_corners
from gsamllavanav.maps.gsam_map import GSamMap, GSamParams
from gsamllavanav import vlmodel
from gsamllavanav import som


def goal_selection_gdino(
    args: ExperimentArgs,
    pred_goal_logs: dict[EpisodeID, list[Point2D]]
) -> dict[EpisodeID, Pose4D]:
    
    cropclient.load_image_cache(alt_env=args.alt_env)
    objects = get_city_refer_objects()
    gsam_params = GSamParams(
        args.gsam_use_segmentation_mask, args.gsam_use_bbox_confidence,
        args.gsam_box_threshold, args.gsam_text_threshold,
        args.gsam_max_box_size, args.gsam_max_box_area,
    )

    predicted_positions = {}  
    for (map_name, obj_id, desc_id), pred_positions in tqdm(pred_goal_logs.items(), desc='selecting target bbox', unit='trajectory'):
        
        final_pred_pose = Pose4D(*pred_positions[-1], args.altitude + GROUND_LEVEL[map_name], 0)
        rgb = cropclient.crop_image(map_name, final_pred_pose, (int(args.altitude*10), int(args.altitude*10)), 'rgb')

        target_object = objects[map_name][obj_id]
        target_name = target_object.processed_descriptions[desc_id].target
        
        gsam_map = GSamMap(map_name, (240, 240), 240/410, [target_name], gsam_params)
        gsam_map.update_observation(final_pred_pose, rgb)
        pred_pos = bbox_corners_to_position(gsam_map.max_confidence_bbox, gsam_map.ground_level)

        camera_z = GROUND_LEVEL[map_name] + args.altitude
        camera_pose = Pose4D(pred_pos.x, pred_pos.y, camera_z, 0)
        depth = cropclient.crop_image(map_name, camera_pose, (100, 100), 'depth')
        z_around_center = camera_pose.z - depth[45:55, 45:55].mean()
        final_pose = Pose4D(pred_pos.x, pred_pos.y, z_around_center + 5, 0)

        predicted_positions[(map_name, obj_id, desc_id)] = final_pose
    
    return predicted_positions


def goal_selection_llava(
    args: ExperimentArgs,
    pred_goal_logs: dict[EpisodeID, list[Point2D]]
) -> dict[EpisodeID, Pose4D]:
    
    NOT_IN_IMAGE = -1
    INVALID_RESPONSE = -2

    vlmodel.load_model('llava-v1.6-34b')
    som.load_model('semantic-sam')
    cropclient.load_image_cache(alt_env=args.alt_env)
    objects = get_city_refer_objects()

    predicted_positions = {}
    for (map_name, obj_id, desc_id), pred_goals in tqdm(pred_goal_logs.items(), desc='selecting target bbox', unit='trajectory'):

        (x, y), yaw = pred_goals[-1], 0.
        ground_level = GROUND_LEVEL[map_name]
        target_object = objects[map_name][obj_id]
        camera_pose = Pose4D(x, y, args.altitude + ground_level, yaw)
        
        rgb = cropclient.crop_image(map_name, camera_pose, (args.altitude*10, args.altitude*10), 'rgb')
        annotated_rgb, masks = som.annotate(rgb, 'semantic-sam', [4])

        prompt = f"Answer the label number of the object that the following text describes. If the ojbect is not present in the image, answer {NOT_IN_IMAGE} instead."
        prompt += f":\n{target_object.descriptions[desc_id]}"
        response = vlmodel.query(annotated_rgb, prompt)

        try:
            label = int(response)
        except ValueError:
            label = INVALID_RESPONSE
        
        bbox_corners = crwh_to_global_bbox(masks[label - 1]['bbox'], rgb.shape[:2], camera_pose, ground_level) if 0 < label <= len(masks) else view_area_corners(camera_pose, ground_level)
        pred_pos = bbox_corners_to_position(bbox_corners, ground_level) if 0 < label <= len(masks) else camera_pose.xyz

        camera_z = GROUND_LEVEL[map_name] + args.altitude
        camera_pose = Pose4D(pred_pos.x, pred_pos.y, camera_z, 0)
        depth = cropclient.crop_image(map_name, camera_pose, (100, 100), 'depth')
        z_around_center = camera_pose.z - depth[45:55, 45:55].mean()
        final_pose = Pose4D(pred_pos.x, pred_pos.y, z_around_center + 5, 0)

        predicted_positions[(map_name, obj_id, desc_id)] = final_pose
    
    return predicted_positions