#!/bin/bash
#SBATCH --job-name="per-class-scale-slam-4"
#SBATCH --nodelist=node[12,13,14,15,16,17]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2,VRAM:48G
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/stud/petp/storage/slurm/logs/slurm-per-class-scale-slam-4-%j.out
#SBATCH --error=/usr/stud/petp/storage/slurm/logs/slurm-per-class-scale-slam-4-%j.out
srun python /usr/stud/petp/code/thesis/iebins/train.py /usr/stud/petp/code/thesis/configs/per_class/per_class_scale_slam_4.txt
