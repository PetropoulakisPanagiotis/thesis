#!/bin/bash
#SBATCH --job-name="per-instance-scale-scannet-unc-2"
#SBATCH --nodelist=node[11,12,14,15,16,17]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2,VRAM:48G
#SBATCH --mem=24G
#SBATCH --time=80:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/stud/petp/storage/slurm/logs/slurm-per-instance-scale-scannet-unc-2-%j.out
#SBATCH --error=/usr/stud/petp/storage/slurm/logs/slurm-per-instance-scale-scannet-unc-2-%j.out
srun python /usr/stud/petp/code/thesis/iebins/train.py /usr/stud/petp/code/thesis/configs/per_instance/per_instance_scale_scannet_unc_2.txt

# SBATCH --nodelist=node[12,14,15,16,17,18]
# SBATCH --nodelist=node[11,12,13,16,17,18]
# 16, 17 
# 11, 12, 13, 14, 15, 18
# SBATCH --nodelist=node[11,12,13,14,15,16,17,18]
# SBATCH --exclude=node1,node2,node3,node4,node5,node6,node7,node8,node9,node10
# SBATCH --gres=gpu:2,VRAM:48G
# SBATCH --gres=gpu:1,VRAM:32G
