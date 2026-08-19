
# [ECCV 2026] PixelU: A U-Shaped Transformer for Efficient End-to-End Pixel Diffusion

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/2606.27760)

<p align="center">
  <img src="asset/visual.jpg" width="100%">
</p>

This repository contains the official PyTorch implementation for **PixelU**, a minimalist, single-stage U-shaped Diffusion Transformer tailored for pixel space. By synergizing zero-cost skip connections with spatial down-sampling, PixelU achieves elegant frequency decoupling. This streamlined design completely eliminates the need for heavyweight pixel decoders, establishing a highly compute-friendly paradigm for end-to-end high-resolution image synthesis.

## 📊 Dataset
Download the [ImageNet](http://image-net.org/download) dataset and place it in your designated `IMAGENET_PATH`. 
Additionally, you will need a pre-trained encoder (e.g., [DINOv2](https://huggingface.co/facebook/dinov2-with-registers-base)) for Representation Alignment (REPA), whose path should be set to `ENCODER_PATH`.

## 🛠️ Installation

A suitable conda environment named `PixelU` can be created and activated using the provided configuration:

```bash
conda env create -f environment.yaml
conda activate PixelU
```

## 🚀 Training

The training entry point is `main.py`. The following training scripts have been tested on 8 GPUs (e.g., NVIDIA B200). Please ensure that your environment variable paths are set correctly before running. Below is an example script for training `PixelU-H-16` on ImageNet 256x256 for 600 epochs:

```bash
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
```

## 📊 Inference / Evaluation

To sample images and evaluate a pre-trained model, use `main.py` with the `--evaluate_gen` flag. Here is an example for evaluating a trained `PixelU-H-16` model at 256x256 resolution:

```bash
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
```

## 🔍 Important Hyperparameters

Here is a breakdown of the key hyperparameters used in the scripts:

* **`--model`**: The name of the model architecture to train or evaluate (e.g., `PixelU-H-32`, `PixelU-H-16`).
* **`--proj_dropout`**: For smaller models (e.g., B and L), this is typically set to `0.0`. For large models (e.g., H), it is recommended to set it to `0.1` to prevent overfitting.
* **`--img_size`**: The target image resolution. Set to `256` or `512` based on the model variant.
* **`--noise_scale`**: The scaling factor for noise. For 256x256 resolution, use `1.0`; for 512x512 resolution, use `2.0`.
* **`--cfg`**: The Classifier-Free Guidance factor for sampling. For example, `2.2` is used for `PixelU-H-16`, and `2.3` is used for `PixelU-H-32`.
* **`--repa_weight`**: The loss weight applied to Representation Alignment (REPA) during training. The default value used in training scripts is `0.01`.
* **`--encoder_path`**: The file path pointing to the pre-trained feature encoder (like DINOv2) needed for the REPA loss computation.
* **`--evaluate_gen`**: A boolean flag that, when present, switches the pipeline from training mode to image generation and evaluation mode.

## ✨ Acknowledgements
Our codebase is adapted from [JiT](https://github.com/LTH14/JiT), [RAE](https://github.com/bytetriper/RAE) and [U-REPA](https://github.com/YuchuanTian/U-REPA). We thank the authors for releasing their code.

## 📝 BibTeX & Citation
If you find our paper and code useful in your research, please consider giving a star and citation.

```BibTeX
@article{guo2026pixelu,
  title={PixelU: A U-Shaped Transformer for Efficient End-to-End Pixel Diffusion},
  author={Guo, Zipeng and Ma, Lichen and He, Yu and Fu, Xiaolong and Fu, Jingling and Huang, Junshi and Li, Yan},
  journal={arXiv preprint arXiv:2606.27760},
  year={2026}
}

