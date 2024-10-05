import gc
import re
from typing import Literal

import numpy as np
import torch
from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image


MODEL_NAME = Literal[
    "llava-v1.6-vicuna-7b",
    "llava-v1.6-vicuna-13b",
    "llava-v1.6-mistral-7b",
    "llava-v1.6-34b",
    "llava-v1.5-7b",
    "llava-v1.5-13b",
    "llava-v1.5-7b-lora",
    "llava-v1.5-13b-lora",
]


_tokenizer = None
_model = None
_image_processor = None


def load_model(model_name: MODEL_NAME):
    
    global _tokenizer, _model, _image_processor

    model_path = f"liuhaotian/{model_name}"

    _tokenizer, _model, _image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        load_4bit=True,
        model_name=get_model_name_from_path(model_path)
    )


def query(
    image: np.ndarray,
    query: str,
    temperature=0,
    top_p=None,
    num_beams=1,
    max_new_tokens=512,
):
    global _tokenizer, _model, _image_processor

    disable_torch_init()

    images = [Image.fromarray(image)]

    image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN if _model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
    query = re.sub(IMAGE_PLACEHOLDER, image_token, query) if IMAGE_PLACEHOLDER in query else image_token + "\n" + query

    conv = conv_templates["chatml_direct"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images_tensor = process_images(images, _image_processor, _model.config).to(_model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, _tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = _model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=[img.size for img in images],
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=None,
        )

    outputs = _tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def unload_model():

    if _model is not None:
        del _tokenizer
        del _model
        del _image_processor
        gc.collect()
        torch.cuda.empty_cache()
    
