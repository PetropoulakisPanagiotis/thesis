#!/bin/bash
#SBATCH --job-name="iebins-3"
#SBATCH --nodelist=node[11,12,13,14,15]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2,VRAM:48G
#SBATCH --mem=24G
#SBATCH --time=50:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/stud/petp/storage/slurm/logs/slurm-iebins-3-%j.out
#SBATCH --error=/usr/stud/petp/storage/slurm/logs/slurm-iebins-3-%j.out
srun python /usr/stud/petp/code/thesis/iebins/train.py /usr/stud/petp/code/thesis/configs/iebins/iebins_3.txt
