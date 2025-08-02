#!/usr/bin/env bash
set -e

pip install vllm==0.9.0.1 transformers==4.53.3

python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-AWQ \
    --host 0.0.0.0 \
    --port 30000 \
    --random-seed 1337 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --context-length 16384 \
    --reasoning-parser qwen3
