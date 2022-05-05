#!/bin/bash
lr_lst=(.001 0.01 0.1)

for lr in "${lr_lst[@]}"; do
    python main.py --method resnet --display_step 25 --lr $lr --save_cp
done