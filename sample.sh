# Evaluation script for PixelU-H-16
export MODEL_NAME="PixelU-H-16"
export OUTPUT_DIR="result/${MODEL_NAME}"
export IMAGENET_PATH="/path/to/your/ImageNet2012"
export ENCODER_PATH="/path/to/dinov2-with-registers-base"
export CKPT_DIR="/path/to/checkpoint-last.pth"

torchrun --nproc_per_node=8 --master_port=29502 \
 main.py \
 --model ${MODEL_NAME} \
 --img_size 256 --noise_scale 1.0 \
 --gen_bsz 512 --num_images 50000 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
 --output_dir ${OUTPUT_DIR} --resume ${CKPT_DIR} \
 --data_path ${IMAGENET_PATH} \
 --encoder_path ${ENCODER_PATH} \
 --evaluate_gen