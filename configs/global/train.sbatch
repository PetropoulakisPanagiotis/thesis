#!/bin/bash
#SBATCH --job-name="global-scale-log-bins-silog-2"
#SBATCH --nodelist=node[11,15,17]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2,VRAM:48G
#SBATCH --mem=24G
#SBATCH --time=50:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/stud/petp/storage/slurm/logs/slurm-global-scale-log-bins-silog-2-%j.out
#SBATCH --error=/usr/stud/petp/storage/slurm/logs/slurm-global-scale-log-bins-silog-2-%j.out
srun python /usr/stud/petp/code/thesis/iebins/train.py /usr/stud/petp/code/thesis/configs/global/global_scale_log_bins_silog_2.txt
