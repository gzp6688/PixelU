# Training script for PixelU-H-16
export MODEL_NAME="PixelU-H-16"
export OUTPUT_DIR="result/${MODEL_NAME}"
export IMAGENET_PATH="/path/to/your/ImageNet2012"
export ENCODER_PATH="/path/to/dinov2-with-registers-base"

torchrun --nproc_per_node=8 --master_port=29502 \
 main.py \
 --model ${MODEL_NAME} \
 --proj_dropout 0.1 \
 --P_mean -0.8 --P_std 0.8 \
 --img_size 256 --noise_scale 1.0 \
 --batch_size 128 --blr 5e-5 \
 --epochs 600 --warmup_epochs 5 \
 --gen_bsz 256 --num_images 50000 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
 --output_dir ${OUTPUT_DIR} \
 --encoder_path ${ENCODER_PATH} \
 --data_path ${IMAGENET_PATH} \
 --eval_freq 80 --online_eval \
 --repa_weight 0.01



# # Training script for PixelU-B-16
# export MODEL_NAME="PixelU-B-16"
# export OUTPUT_DIR="result/${MODEL_NAME}"
# export IMAGENET_PATH="/path/to/your/ImageNet2012"
# export ENCODER_PATH="/path/to/dinov2-with-registers-base"

# torchrun --nproc_per_node=8 --master_port=29502 \
#  main.py \
#  --model ${MODEL_NAME} \
#  --proj_dropout 0.0 \
#  --P_mean -0.8 --P_std 0.8 \
#  --img_size 256 --noise_scale 1.0 \
#  --batch_size 128 --blr 5e-5 \
#  --epochs 600 --warmup_epochs 5 \
#  --gen_bsz 256 --num_images 50000 --cfg 2.7 --interval_min 0.1 --interval_max 1.0 \
#  --output_dir ${OUTPUT_DIR} \
#  --encoder_path ${ENCODER_PATH} \
#  --data_path ${IMAGENET_PATH} \
#  --eval_freq 80 --online_eval \
#  --repa_weight 0.01



# # Training script for PixelU-L-16
# export MODEL_NAME="PixelU-L-16"
# export OUTPUT_DIR="result/${MODEL_NAME}"
# export IMAGENET_PATH="/path/to/your/ImageNet2012"
# export ENCODER_PATH="/path/to/dinov2-with-registers-base"

# torchrun --nproc_per_node=8 --master_port=29502 \
#  main.py \
#  --model ${MODEL_NAME} \
#  --proj_dropout 0.0 \
#  --P_mean -0.8 --P_std 0.8 \
#  --img_size 256 --noise_scale 1.0 \
#  --batch_size 128 --blr 5e-5 \
#  --epochs 600 --warmup_epochs 5 \
#  --gen_bsz 256 --num_images 50000 --cfg 2.4 --interval_min 0.1 --interval_max 1.0 \
#  --output_dir ${OUTPUT_DIR} \
#  --encoder_path ${ENCODER_PATH} \
#  --data_path ${IMAGENET_PATH} \
#  --eval_freq 80 --online_eval \
#  --repa_weight 0.01



# # Training script for PixelU-H-32
# export MODEL_NAME="PixelU-H-32"
# export OUTPUT_DIR="result/${MODEL_NAME}"
# export IMAGENET_PATH="/path/to/your/ImageNet2012"
# export ENCODER_PATH="/path/to/dinov2-with-registers-base"

# torchrun --nproc_per_node=8 --master_port=29502 \
#  main.py \
#  --model ${MODEL_NAME} \
#  --proj_dropout 0.1 \
#  --P_mean -0.8 --P_std 0.8 \
#  --img_size 512 --noise_scale 2.0 \
#  --batch_size 128 --blr 5e-5 \
#  --epochs 600 --warmup_epochs 5 \
#  --gen_bsz 256 --num_images 50000 --cfg 2.3 --interval_min 0.1 --interval_max 1.0 \
#  --output_dir ${OUTPUT_DIR} \
#  --encoder_path ${ENCODER_PATH} \
#  --data_path ${IMAGENET_PATH} \
#  --eval_freq 80 --online_eval \
#  --repa_weight 0.01