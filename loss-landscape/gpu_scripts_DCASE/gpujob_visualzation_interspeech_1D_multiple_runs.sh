#!/usr/bin/env bash
#SBATCH --job-name=vis_nsee
#SBATCH --time=72:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=./gpu_scripts_DCASE/logs/slurm_nseeds_interspeech_visualization.out

# This activates the environment (given .envrc is in the current directory)
direnv allow . && eval "\$(direnv export bash)"

mpirun -n 1 python plot_surface_folder_loop.py --cuda --partition train --dataset dcase --x=-1:1:11 --dir_type states --data-root /data/eihw-gpu5/milliman/DCASE/DCASE2020/metadata/ --features /data/eihw-gpu5/milliman/DCASE/DCASE2020/mel_spectrograms/features.csv --model_folder /nas/staff/data_work/manuel/cloned_repos/visualisation/loss-landscape/model_selection_interspeech_paper/01_interspeech_multiseed_1D/RQ1_Generalisation/ --xnorm filter --xignore biasbn --ngpu 1 --plot --n_seeds 10 --random_seed 42 --loss_max 5
