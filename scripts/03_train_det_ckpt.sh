#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`

NUM_GPUS=1

python ./train.py --plugins='replay' --model_dir='./logs/exp1_fusion_C3SM_spec' --checkpoint_at=-1