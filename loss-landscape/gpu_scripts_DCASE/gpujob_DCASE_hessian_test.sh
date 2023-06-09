#!/usr/bin/env bash
#SBATCH --job-name=DCASEvh1
#SBATCH --time=72:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=./gpu_scripts_DCASE/logs/slurm_hessian_test.out

# This activates the environment (given .envrc is in the current directory)
direnv allow . && eval "\$(direnv export bash)"

mpirun -n 1 python plot_hessian_eigen.py --cuda --mpi --model cnn14 --partition test --dataset dcase --x=-1.5:1.8:21 --y=-2:2:21 --dir_type weights --data-root /data/eihw-gpu5/milliman/DCASE/DCASE2020/metadata/ --features /data/eihw-gpu5/milliman/DCASE/DCASE2020/mel_spectrograms/features.csv --model_file /nas/staff/data_work/manuel/cloned_repos/visualisation/loss-landscape/DCASE2020/trained_nets/run_batch32_cnn14_Adam/test_hessian/state.pth.tar --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --ngpu 1 --plot --batch_size 2
