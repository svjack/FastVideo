#!/bin/bash

# 定义提示词内容
prompts=(
    "An orange tabby cat with green eyes notices a pure white long-eared rabbit shivering in the rain. The tabby cat tilts its head curiously, gently lifting the rabbit with its mouth, showing both curiosity and protectiveness."
    "The orange tabby cat carefully carries the pure white rabbit into a cozy wooden house, placing it on a soft blanket near the fireplace, its green eyes filled with concern."
    "The orange tabby cat skillfully slices carrots with its paws in a small kitchen, while the pure white rabbit watches from the blanket, its long ears twitching with anticipation."
    "The orange tabby cat gently places a bowl of chopped carrots before the pure white rabbit, purring softly as the rabbit begins to eat, its green eyes reflecting gratitude."
    "The orange tabby cat playfully bats a ball of yarn, while the pure white rabbit hops around excitedly, their bond strengthening in this joyful interaction."
    "The orange tabby cat and pure white rabbit settle together on a cozy blanket by the fireplace, the cat wrapping its tail around the rabbit as they drift into sleep, warmed by the flickering flames."
)

# 设置提示词文件路径
prompt_file="./data/prompts/cat_and_rabbit_story_prompts.txt"
mkdir -p "$(dirname "$prompt_file")"  # 创建目录（如果不存在）
> "$prompt_file"  # 清空文件内容

# 将所有提示词写入同一个文件
for prompt in "${prompts[@]}"; do
    echo "$prompt" >> "$prompt_file"
done

echo "所有提示词已成功写入 $prompt_file 文件。"

# 设置视频生成脚本的参数
num_gpus=1
export MODEL_BASE="data/FastHunyuan-diffusers"
output_dir="./outputs/videos/cat_and_rabbit_story"
mkdir -p "$output_dir"  # 创建输出目录（如果不存在）

# 运行视频生成脚本
echo "生成视频..."
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 12345 \
    fastvideo/sample/sample_t2v_hunyuan_hf.py \
    --height 720 \
    --width 1280 \
    --num_frames 45 \
    --num_inference_steps 6 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 17 \
    --prompt "$prompt_file" \
    --seed 1024 \
    --output_path "$output_dir" \
    --model_path "$MODEL_BASE" \
    --quantization "nf4" \
    --cpu_offload

echo "视频生成完成并保存到 $output_dir。"