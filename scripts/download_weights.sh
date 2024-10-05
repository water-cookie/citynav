mkdir -p weights/groundingdino
wget -P weights/groundingdino https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -P weights/groundingdino https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

mkdir -p weights/mobile_sam
wget -P weights/mobile_sam https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

mkdir -p weights/som
wget -P weights/som https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth
wget -P weights/som https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt
wget -P weights/som https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

mkdir -p weights/vlnce
cd weights/vlnce
wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models.zip &&
unzip ddppo-models.zip &&
rm ddppo-models.zip
gdown https://drive.google.com/uc?id=1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr &&
unzip R2R_VLNCE_v1-3_preprocessed.zip &&
rm R2R_VLNCE_v1-3_preprocessed.zip
cd ../..