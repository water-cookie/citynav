CUDA_VISIBLE_DEVICES='0'

cd ..

python main_goal_predictor.py \
    --mode train \
    --model cma_with_map \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --gsam_use_map_cache \
    --learning_rate 0.0015 \
    --train_batch_size 12 \
    --train_trajectory_type sp \
    --eval_max_timestep 15 \
    --log

python main_goal_predictor.py \
    --mode train \
    --model cma_with_map \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --gsam_use_map_cache \
    --learning_rate 0.0015 \
    --train_batch_size 12 \
    --train_trajectory_type mturk \
    --eval_max_timestep 15 \
    --log