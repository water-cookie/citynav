CUDA_VISIBLE_DEVICES='0'

cd ..

python main_goal_predictor.py \
    --mode eval \
    --model cma_with_map \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --eval_batch_size 50 \
    --eval_max_timestep 15 \
    --gsam_use_map_cache \
    --checkpoint checkpoints/baseline_with_map/sp_50.0_0.2/002.pth

python main_goal_predictor.py \
    --mode eval \
    --model cma_with_map \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --eval_batch_size 50 \
    --eval_max_timestep 15 \
    --gsam_use_map_cache \
    --checkpoint checkpoints/baseline_with_map/cma_with_map/mturk_50.0_0.2/000.pth

