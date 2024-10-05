CUDA_VISIBLE_DEVICES='0'

cd ..

python main_vlnce.py \
    --mode train \
    --learning_rate 0.001 \
    --train_batch_size 8 \
    --epochs 5 \
    --policy cma \
    --use_progress_monitor \
    --train_trajectory_type sp
