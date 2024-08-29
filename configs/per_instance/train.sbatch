#!/bin/bash
#SBATCH --job-name="per-instance-scale-slam-3"
#SBATCH --nodelist=node[11,13,14,15,16]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2,VRAM:48G
#SBATCH --mem=24G
#SBATCH --time=80:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/stud/petp/storage/slurm/logs/slurm-per-instance-scale-slam-3-%j.out
#SBATCH --error=/usr/stud/petp/storage/slurm/logs/slurm-per-instance-scale-slam-3-%j.out
srun python /usr/stud/petp/code/thesis/iebins/train.py /usr/stud/petp/code/thesis/configs/per_instance/per_instance_scale_slam_3.txt
