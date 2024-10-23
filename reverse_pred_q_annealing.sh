#!/bin/bash

python3 reverse_pred_q_annealing.py \
--checkpoint_dir './result/test' \
--target_properties '{"QED": 0.9, "SAS": 2.0}' \
--bit_size_per_constant 8 \
--emb_size 512 \
--penalty 0.0 \
--iter 10 \
--n_sa_sampling 1 \
--method 'qa'
