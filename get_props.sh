#!/bin/bash

python3 get_props.py \
--dataset_dir './dataset' \
--train_size 1000000 \
--test_size 100000 \
--save_dir './result/test' \
--seed 1
