#!/usr/bin/env bash
#SBATCH --job-name=dcaseG1
#SBATCH --time=72:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=./gpu_scripts/logs/slurm_01_test.out

# This activates the environment (given .envrc is in the current directory)
direnv allow . && eval "\$(direnv export bash)"

python Simon_Code/benchrunner.py --data-root /data/eihw-gpu5/milliman/DCASE/DCASE2020/metadata --features /data/eihw-gpu5/milliman/DCASE/DCASE2020/mel_spectrograms/features.csv --feature_dir /data/eihw-gpu5/milliman/DCASE/DCASE2020/mel_spectrograms/ --results-root /nas/staff/data_work/manuel/cloned_repos/visualisation/results --device cuda --num_gpus 1
