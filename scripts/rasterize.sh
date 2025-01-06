conda create --yes --name rasterize --channel conda-forge pdal python=3.10 &&
conda activate rasterize &&
pip install rasterio pillow &&
python rasterize.py "$1" &&
conda deactivate