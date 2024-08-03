#!/bin/bash
#SBATCH --job-name="global-scale-canonical-log-bins-100-l1-1"
#SBATCH --nodelist=node[11,12,13,14,15,16,17]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2,VRAM:48G
#SBATCH --mem=24G
#SBATCH --time=50:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/stud/petp/storage/slurm/logs/slurm-global-scale-canonical-log-bins-100-l1-1-%j.out
#SBATCH --error=/usr/stud/petp/storage/slurm/logs/slurm-global-scale-canonical-log-bins-100-l1-1-%j.out
srun python /usr/stud/petp/code/thesis/iebins/train.py /usr/stud/petp/code/thesis/configs/global/global_scale_canonical_log_bins_100_l1_1.txt
