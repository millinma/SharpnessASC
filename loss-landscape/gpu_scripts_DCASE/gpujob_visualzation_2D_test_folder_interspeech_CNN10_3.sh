#!/usr/bin/env bash
#SBATCH --job-name=vi_c10_3
#SBATCH --time=72:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=./gpu_scripts_DCASE/logs/slurm_CNN10_3_interspeech_visualization_1D_folder.out

# This activates the environment (given .envrc is in the current directory)
direnv allow . && eval "\$(direnv export bash)"

mpirun -n 1 python plot_surface_folder.py --cuda --partition train --dataset dcase --x=-0.25:0.25:11 --y=-0.25:0.25:11 --dir_type weights --data-root /data/eihw-gpu5/milliman/DCASE/DCASE2020/metadata/ --features /data/eihw-gpu5/milliman/DCASE/DCASE2020/mel_spectrograms/features.csv --model_folder /nas/staff/data_work/manuel/cloned_repos/visualisation/loss-landscape/all_grid_interspeech/03_seed_2D_visualisations/CNN10/ --xnorm filter --xignore biasbn --ngpu 1 --plot --random_seed=44 --loss_max 20 --ynorm filter --yignore biasbn
