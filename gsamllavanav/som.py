import gc
from PIL import Image
from typing import Literal

import numpy as np
import torch

# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import inference_seem_pano

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam.architectures.build import build_model as build_model_semsam
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto

from gsamllavanav.defaultpaths import SEMSAM_CFG, SEEM_CFG, SEMSAM_CHECKPOINT_PATH, SAM_CHECKPOINT_PATH, SEEM_CHECKPOINT_PATH

# options
OPT_SEMSAM = load_opt_from_config_file(SEMSAM_CFG)
OPT_SEEM = load_opt_from_config_file(SEEM_CFG)
OPT_SEEM = init_distributed_seem(OPT_SEEM)


ModelName = Literal['seem', 'semantic-sam', 'sam']
models = {}


def load_model(name: ModelName):
    if name == 'semantic-sam':
        models[name] = BaseModel(OPT_SEMSAM, build_model_semsam(OPT_SEMSAM)).from_pretrained(SEMSAM_CHECKPOINT_PATH).eval().cuda()
    if name == 'sam':
        models[name] = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).eval().cuda()
    if name == 'seem':
        models[name] = BaseModel_Seem(OPT_SEEM, build_model_seem(OPT_SEEM)).from_pretrained(SEEM_CHECKPOINT_PATH).eval().cuda()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                models[name].model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)


@torch.no_grad()
def annotate(
    image: np.ndarray,
    name: ModelName = 'semantic-sam',
    level: list[int] = [4],
    mask_transparency: float = 0.,
    label_mode: Literal['a', '1'] = '1',
    anno_mode: list[str] = ["Mark"],
    *args,
    **kwargs,
):
    '''
    Parameters
    ----------
    level: for semantic-sam: subset of [1, 2, 3, 4, 5, 6]
    mask_transparency: alpha value of segmentation amsk
    anno_mode: combo of ["Mask", "Box", "Mark"]
    '''
    image = Image.fromarray(image)

    text_size, hole_scale, island_scale=500,100,100
    text, text_part, text_thresh = '','','0.0'

    if name not in models:
        load_model(name)

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        try:
            if name == 'semantic-sam':
                output, mask = inference_semsam_m2m_auto(models[name], image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic=False, label_mode=label_mode, alpha=mask_transparency, anno_mode=anno_mode, *args, **kwargs)
            if name == 'sam':
                output, mask = inference_sam_m2m_auto(models[name], image, text_size, label_mode, mask_transparency, anno_mode)
            if name == 'seem':
                output, mask = inference_seem_pano(models[name], image, text_size, label_mode, mask_transparency, anno_mode)
        except (UnboundLocalError, RuntimeError):
                output, mask = np.array(image), []
        return output, mask


def unload_model(name: ModelName):
    if name not in models:
        return False

    del models[name]
    gc.collect()
    torch.cuda.empty_cache()

    return True
