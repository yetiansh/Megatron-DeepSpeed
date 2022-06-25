#!/bin/bash

mkdir -p /Megatron-DeepSpeed/data
cd /Megatron-DeepSpeed/data
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

mkdir -p /Megatron-DeepSpeed/checkpoints/gpt2_345m
cd /Megatron-DeepSpeed/checkpoints/gpt2_345m
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
unzip megatron_lm_345m_v0.0.zip
rm megatron_lm_345m_v0.0.zip