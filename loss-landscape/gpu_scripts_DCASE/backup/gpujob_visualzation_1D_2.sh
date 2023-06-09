#!/usr/bin/env bash
#SBATCH --job-name=dcase_v2
#SBATCH --time=72:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=./gpu_scripts_DCASE/logs/slurm_visualization_1D_2.out

# This activates the environment (given .envrc is in the current directory)
direnv allow . && eval "\$(direnv export bash)"

mpirun -n 1 python plot_surface.py --cuda --model cnn10 --partition train --dataset dcase --x=-1:1:101 --dir_type weights --data-root /data/eihw-gpu5/milliman/DCASE/DCASE2020/metadata/ --features /data/eihw-gpu5/milliman/DCASE/DCASE2020/mel_spectrograms/features.csv --model_file /nas/staff/data_work/manuel/cloned_repos/visualisation/loss-landscape/model_selection_interspeech_paper/first_test/cnn10_None_KFACOptimizer_1e-05_16_50_42_None/state.pth.tar --xnorm filter --xignore biasbn --ngpu 1 --plot 
