#!/bin/bash

hidden_channels_lst=(16 32 64 128 256)
dropout_lst=(0 .5)

for hidden_channels in "${hidden_channels_lst[@]}"; do
    for dropout in "${dropout_lst[@]}"; do
        python main.py --method cnn --dropout $dropout --hidden_channels $hidden_channels --display_step 25
    done
done