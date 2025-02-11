from pathlib import Path


PROJECT_ROOT = Path(".")

CITYREFER_DATA_DIR = PROJECT_ROOT/"data/cityrefer"
OBJECTS_PATH = CITYREFER_DATA_DIR/"objects.json"
PROCESSED_DECRIPTIONS_PATH = CITYREFER_DATA_DIR/"processed_descriptions.json"
MTURK_TRAJECTORY_DIR = PROJECT_ROOT/"data/citynav"

ORTHO_IMAGE_DIR = PROJECT_ROOT/"data/rgbd"

CHECKPOINT_DIR = PROJECT_ROOT/"checkpoints/vlnce"

WEIGHTS_DIR = PROJECT_ROOT/"weights/vlnce"
DEPTH_ENCODER_WEIGHT_PATH = WEIGHTS_DIR/"data/ddppo-models/gibson-2plus-resnet50.pth"
WORD_EMBEDDING_PATH = WEIGHTS_DIR/"R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz"
