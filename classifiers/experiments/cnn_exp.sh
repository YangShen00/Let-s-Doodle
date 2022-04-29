#!/bin/bash

hidden_channels_lst=(16 32 64 128)
dropout_lst=(0 .5)

for hidden_channels in "${hidden_channels_lst[@]}"; do
    for dropout1 in "${dropout_lst[@]}"; do
        for dropout2 in "${dropout_lst[@]}"; do
            python3 main.py --method cnn --dropout1 $dropout1 --dropout2 $dropout2 --hidden_channels $hidden_channels --display_step 25
            --SGD --lr 0.001
        done
    done
done