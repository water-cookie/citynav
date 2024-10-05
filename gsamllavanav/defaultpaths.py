from pathlib import Path


PROJECT_ROOT = Path(".")

WEIGHTS_DIR = PROJECT_ROOT/"weights"
GOAL_PREDICTOR_CHECKPOINT_DIR = PROJECT_ROOT/"checkpoints/goal_predictor"
BASELINE_WITH_MAP_CHECKPOINT_DIR = PROJECT_ROOT/"checkpoints/baseline_with_map"

CITYREFER_DATA_DIR = PROJECT_ROOT/"data/cityrefer"
OBJECTS_PATH = CITYREFER_DATA_DIR/"objects.json"
PROCESSED_DECRIPTIONS_PATH = CITYREFER_DATA_DIR/"processed_descriptions.json"
MTURK_TRAJECTORY_DIR = PROJECT_ROOT/"data/citynav"

ORTHO_IMAGE_DIR = PROJECT_ROOT/"data/rgbd/ortho_projection_images"
SUBBLOCKS_DIR = PROJECT_ROOT/"data/subblocks"

# SoM
## configs
SOM_WEIGHTS_DIR = WEIGHTS_DIR/"som"
SOM_CONFIG_DIR = PROJECT_ROOT/'SoM/configs'
SEMSAM_CFG = SOM_CONFIG_DIR/"semantic_sam_only_sa-1b_swinL.yaml"
SEEM_CFG = SOM_CONFIG_DIR/"seem_focall_unicl_lang_v1.yaml"
## weights
SEMSAM_CHECKPOINT_PATH = SOM_WEIGHTS_DIR/"swinl_only_sam_many2many.pth"
SAM_CHECKPOINT_PATH = SOM_WEIGHTS_DIR/"sam_vit_h_4b8939.pth"
SEEM_CHECKPOINT_PATH = SOM_WEIGHTS_DIR/"seem_focall_v1.pt"

# GSAM
GDINO_MODEL_SIZE = "B" if False else "T"
## configs
GDINO_CONFIG_DIR = PROJECT_ROOT/"gsamllavanav/configs/groundingdino"
GDINO_CONFIG_PATH = GDINO_CONFIG_DIR/f"GroundingDINO_Swin{'B_cfg' if GDINO_MODEL_SIZE == 'B' else 'T_OGC'}.py"
## weights
GDINO_CHECKPOINT_PATH = WEIGHTS_DIR/f"groundingdino/groundingdino_swin{'b_cogcoor' if GDINO_MODEL_SIZE == 'B' else 't_ogc'}.pth"
MOBILE_SAM_CHECKPOINT_PATH = WEIGHTS_DIR/"mobile_sam/mobile_sam.pt"
## data
GSAM_MAPS_DIR = PROJECT_ROOT/"data/gsam"

# Goal Predictor
DEPTH_ENCODER_CHECKPOINT_PATH = WEIGHTS_DIR/"vlnce/data/ddppo-models/gibson-2plus-resnet50.pth"