#!/bin/bash
#SBATCH --job-name="global-scale-canonical-bins-80-silog-2"
#SBATCH --nodelist=node[15,16]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2,VRAM:48G
#SBATCH --mem=24G
#SBATCH --time=50:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/stud/petp/storage/slurm/logs/slurm-global-scale-canonical-bins-80-silog-2-%j.out
#SBATCH --error=/usr/stud/petp/storage/slurm/logs/slurm-global-scale-canonical-bins-80-silog-2-%j.out
srun python /usr/stud/petp/code/thesis/iebins/train.py /usr/stud/petp/code/thesis/configs/global/global_scale_canonical_bins_80_silog_2.txt
