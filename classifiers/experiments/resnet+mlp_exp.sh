#!/bin/bash
lr_lst=(.001 0.01)
num_layers_lst=(2 3)

for lr in "${lr_lst[@]}"; do
    for num_layers in "${num_layers_lst[@]}"; do
        python main.py --method "resnet+mlp" --display_step 25 --num_layers $num_layers --lr $lr --save_cp --epochs 20
    done
done