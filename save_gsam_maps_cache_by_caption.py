import warnings
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from gsamllavanav.cityreferobject import (filter_objects_with_description,
                                          get_city_refer_objects)
from gsamllavanav.defaultpaths import GSAM_MAPS_DIR
from gsamllavanav.maps.gsam_map import GSamMap, GSamParams
from gsamllavanav.observation import cropclient
from gsamllavanav.subblocks import _split_map


ALTITUDE = 50
MAP_SIZE = 240
MAP_METERS = 410

dirname = GSAM_MAPS_DIR/f"full_scan/{(ALTITUDE, MAP_SIZE, MAP_METERS)}"
dirname.mkdir(exist_ok=True)

objects = filter_objects_with_description(get_city_refer_objects())
object_names = defaultdict(set)
for map_name, single_map_objects in tqdm(objects.items(), desc='maps', position=0):
    for obj_id, obj in tqdm(single_map_objects.items(), desc='objects', position=1, leave=False):
        for desc_id, desc in enumerate(obj.processed_descriptions):
            object_names[map_name].add(desc.target)
            object_names[map_name].update(desc.surroundings)


gsam_maps_by_map = defaultdict(dict)
cropclient.load_image_cache()
warnings.filterwarnings('ignore')

def gsam_scan_map(map_name: str, caption: str):
    gsam_map = GSamMap(map_name, (MAP_SIZE, MAP_SIZE), MAP_SIZE/MAP_METERS, [caption], GSamParams(True, 0.2))

    if not caption:
        return gsam_map.to_array(np.float16)

    for pose in _split_map(map_name, ALTITUDE):
        rgb = cropclient.crop_image(map_name, pose, (ALTITUDE*10, ALTITUDE*10), 'rgb')
        if (rgb > 0).mean() > 0.15:
            gsam_map.update_observation(pose, rgb)
    return gsam_map.to_array(np.float16)

for map_name, single_map_names in tqdm(object_names.items(), desc='maps', position=0):
    for caption in tqdm(single_map_names, desc='names', position=1, leave=False):
        caption : str = caption.replace('/', ' ')
        if (filename := dirname/f"{map_name}-{caption}.npy").exists():
            continue
        np.save(filename, gsam_scan_map(map_name, caption))