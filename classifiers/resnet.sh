#!/bin/bash
#SBATCH -J resnet  # Job name
#SBATCH -o resnet.log  # Name of stdout output file (%j expands to jobId)
#SBATCH -e resnet.err  # Name of stderr output file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<netid>@cornell.edu
#SBATCH -N 1  # Total number of CPU nodes requested
#SBATCH -n 4  # Total number of CPU cores requrested
#SBATCH -t 24:00:00  # Run time (hh:mm:ss)
#SBATCH --mem=20000  # CPU Memory pool for all cores
#SBATCH --partition=default_partition --gres=gpu:1
#SBATCH --get-user-env

# Put the command you want to run here. For example:
bash experiments/resnet_exp.sh
