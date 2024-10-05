CUDA_VISIBLE_DEVICES='0'

cd ..

python main_vlnce.py \
    --mode eval \
    --policy cma \
    --checkpoint checkpoints/vlnce/cma_sp/004.pth
