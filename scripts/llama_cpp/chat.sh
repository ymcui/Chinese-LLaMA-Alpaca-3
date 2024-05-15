#!/bin/bash

# script to chat with Llama-3-Chinese-Instruct model
# usage: ./chat.sh llama-3-chinese-instruct-gguf-model-path your-first-instruction
# WARNING: the hyperparameters are not optimal, please tune them yourself

FIRST_INSTRUCTION=$2
SYSTEM_PROMPT="You are a helpful assistant. 你是一个乐于助人的助手。"

./main -m $1 --color -i \
-c 0 -t 6 --temp 0.2 --repeat_penalty 1.1 -ngl 999 \
-r '<|eot_id|>' \
--in-prefix '<|start_header_id|>user<|end_header_id|>\n\n' \
--in-suffix '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' \
-p "<|start_header_id|>system<|end_header_id|>\n\n$SYSTEM_PROMPT<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n$FIRST_INSTRUCTION<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
