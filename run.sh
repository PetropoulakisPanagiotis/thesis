#!/bin/bash
#SBATCH --job-name="My Training"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2,VRAM:12G
#SBATCH --mem=10G
#SBATCH --time=0:05:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python iebins/train.py configs/arguments_train_kittieigen.txt
