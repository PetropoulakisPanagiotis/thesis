#!/bin/bash
#SBATCH --job-name="iebins-scannet"
#SBATCH --nodelist=node[11,12,13,14,15,16,17]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2,VRAM:48G
#SBATCH --mem=24G
#SBATCH --time=50:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/stud/petp/storage/slurm/logs/slurm-iebins-scannet-%j.out
#SBATCH --error=/usr/stud/petp/storage/slurm/logs/slurm-iebins-scannet-%j.out
srun python /usr/stud/petp/code/thesis/iebins/train.py /usr/stud/petp/code/thesis/configs/iebins/iebins_scannet.txt
