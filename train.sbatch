#!/bin/bash
#SBATCH --job-name="depth-pred"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/stud/petp/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/usr/stud/petp/storage/slurm/logs/slurm-%j.out
srun python /usr/stud/petp/code/thesis/iebins/train.py /usr/stud/petp/code/thesis/configs/arguments_train_kittieigen.txt




# SBATCH --exclude=node1,node2,node3,node4,node5,node6,node7,node8,node9,node10
# SBATCH --gres=gpu:2,VRAM:48G
# SBATCH --nodelist=node[11,12,13,14,15,16,17,18]
