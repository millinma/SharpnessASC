#!/usr/bin/env bash
#SBATCH --job-name=vi_rq3_1
#SBATCH --time=72:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=./gpu_scripts_DCASE/logs/slurm_RQ3_1_interspeech_visualization_1D_folder.out

# This activates the environment (given .envrc is in the current directory)
direnv allow . && eval "\$(direnv export bash)"

mpirun -n 1 python plot_surface_folder.py --cuda --partition train --dataset dcase --x=-0.25:0.25:11 --y=-0.25:0.25:11 --dir_type weights --data-root /data/eihw-gpu5/milliman/DCASE/DCASE2020/metadata/ --features /data/eihw-gpu5/milliman/DCASE/DCASE2020/mel_spectrograms/features.csv --model_folder /nas/staff/data_work/manuel/cloned_repos/visualisation/loss-landscape/model_selection_interspeech_paper/01_interspeech_seed_2D/RQ3_UnseenDevices/ --xnorm filter --xignore biasbn --ngpu 1 --plot --random_seed 42 --loss_max 5 --ynorm filter --yignore biasbn
