#!/bin/bash

# 定义提示词内容
prompts=(
    "Against a backdrop of ancient trees shrouded in mist, Wukong stands prominently, his sophisticated black sunglasses adding a modern edge to his mythical appearance. His face, a striking blend of human and simian traits, is characterized by intense eyes behind the dark lenses and dense fur that frames his strong features. The ornate golden armor with swirling patterns shimmers as he crosses his arms across his chest, his posture exuding authority. He nods his head rhythmically, a subtle smile playing on his lips as the sunglasses reflect the diffused light."
    "Through tranquil space with traditional decorations, Wukong holds red envelopes, his stylish sunglasses creating an intriguing blend with his fur-covered face showing generous spirit. His elaborate golden armor adorned with intricate patterns gleams beside lucky packets, his strong features expressing giving joy."
    "Against peaceful light, Wukong examines a bespoke leather journal, his black sunglasses framing his fur-covered face thoughtfully. His elaborate golden armor with intricate patterns gleams as he appreciates craftmanship, his strong simian features showing writer's interest."
    "In misty light among paper-cut designs, Wukong makes a respectful gesture, his sleek sunglasses harmonizing with his fur-covered face showing artistic appreciation. His elaborate golden armor with dragon patterns catches intricate shadows as he shares cultural greetings, his strong simian features radiating tradition."
    "In misty light, Wukong contemplates a chessboard, his fur-covered face showing thoughtful consideration. His elaborate golden armor with intricate patterns gleams as he studies the pieces, his strong features deep in strategic thought."
    "Against a peaceful backdrop decorated with paper-cut designs, Wukong stands with a tray of mandarin oranges, his stylish sunglasses harmonizing with his fur-covered face showing gracious hospitality. His elaborate golden armor adorned with swirling patterns catches the gentle light as he offers the lucky fruit, his strong simian features radiating traditional courtesy."
)

# 设置提示词文件路径
prompt_file="./data/prompts/prompt.txt"
mkdir -p "$(dirname "$prompt_file")"  # 创建目录（如果不存在）
> "$prompt_file"  # 清空文件内容

# 将提示词写入文件
for prompt in "${prompts[@]}"; do
    echo "$prompt" >> "$prompt_file"
done

echo "提示词已成功写入 $prompt_file 文件。"

# 设置视频生成脚本的参数
num_gpus=1
export MODEL_BASE="data/FastHunyuan-diffusers"
output_path="./outputs/videos/hunyuan_quant/nf4/"
mkdir -p "$output_path"  # 创建输出目录（如果不存在）

# 运行视频生成脚本
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
    --output_path "$output_path" \
    --model_path "$MODEL_BASE" \
    --quantization "nf4" \
    --cpu_offload \
    --lora_checkpoint_dir "FastVideo/Hunyuan-Black-Myth-Wukong-lora-weight"