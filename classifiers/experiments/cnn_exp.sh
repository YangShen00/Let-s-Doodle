#!/bin/bash

lr_lst=(0.001 0.002 0.003 0.004 0.005)

for lr in "${lr_lst[@]}"; do
    python3 main.py --method cnn --display_step 25--SGD --lr $lr
done