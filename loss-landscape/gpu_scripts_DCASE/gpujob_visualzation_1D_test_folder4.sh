#!/usr/bin/env bash
#SBATCH --job-name=dcase_v4
#SBATCH --time=72:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=./gpu_scripts_DCASE/logs/slurm_visualization_1D_folder4.out

# This activates the environment (given .envrc is in the current directory)
direnv allow . && eval "\$(direnv export bash)"

mpirun -n 1 python plot_surface_folder.py --cuda --partition train --dataset dcase --x=-0.1:0.1:51 --dir_type states --data-root /data/eihw-gpu5/milliman/DCASE/DCASE2020/metadata/ --features /data/eihw-gpu5/milliman/DCASE/DCASE2020/mel_spectrograms/features.csv --model_folder /nas/staff/data_work/manuel/cloned_repos/visualisation/loss-landscape/model_selection_interspeech_paper/eighth_test/ --xnorm filter --xignore biasbn --ngpu 1 --plot --no_random_seed --loss_max 5
