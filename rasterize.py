import os
from pathlib import Path

import numpy as np
import pdal
import rasterio
from PIL import Image


Z_NODATA = -9999


def generate_pipeline(ply_path: Path):
    return f"""
    [
        "{ply_path.as_posix()}",
        {{
            "type": "writers.gdal",
            "filename": "z.tif",
            "binmode": true,
            "dimension": "Z",
            "data_type": "double",
            "nodata": {Z_NODATA},
            "output_type": "max",
            "resolution": 0.1
        }},
        {{
            "type": "writers.gdal",
            "filename": "r.tif",
            "binmode": true,
            "dimension": "Red",
            "data_type": "uint8",
            "nodata": 0,
            "output_type": "mean",
            "resolution": 0.1
        }},
        {{
            "type": "writers.gdal",
            "filename": "g.tif",
            "binmode": true,
            "dimension": "Green",
            "data_type": "uint8",
            "nodata": 0,
            "output_type": "mean",
            "resolution": 0.1
        }},
        {{
            "type": "writers.gdal",
            "filename": "b.tif",
            "binmode": true,
            "dimension": "Blue",
            "data_type": "uint8",
            "nodata": 0,
            "output_type": "mean",
            "resolution": 0.1
        }}
    ]
    """


def rasterize_ply(ply_path, out_dir='rgbd'):

    ply_path = Path(ply_path)
    block_name = ply_path.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # rasterize the point cloud
    pipeline_json = generate_pipeline(ply_path)
    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()

    # set z nodata to the minimum z value minus one
    with rasterio.open('z.tif') as z_dataset:
        z = z_dataset.read(1)
        min_z = (z != Z_NODATA).min()
        z[z == Z_NODATA] = min_z - 1
        with rasterio.open(out_dir/f'{block_name}.tif', 'w', **z_dataset.profile) as dst:
            dst.write(z, 1)
        
    # write rgb image
    rgb = np.stack([rasterio.open(f"{color}.tif").read(1) for color in ["r", "g", "b"]], axis=-1)
    Image.fromarray(rgb).save(out_dir/f"{block_name}.png")

    # clean up
    for temp_file in ["z.tif", "r.tif", "g.tif", "b.tif"]:
        os.remove(temp_file)


if __name__ == "__main__":
    
    import sys

    ply_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/rgbd'
    for ply_path in Path(ply_dir).rglob("*.ply"):
        if ply_path.stem not in ["cambridge_block_0", "cambridge_block_1", "cambridge_block_34"]:
            rasterize_ply(ply_path, out_dir)
