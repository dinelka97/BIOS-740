#!/bin/bash

#SBATCH --job-name=cnn_test_gpu
#SBATCH --output=cnn_test_%j.out
#SBATCH --error=cnn_test_%j.out
#SBATCH -n 1
#SBATCH --cpus-per-task=50        # number of CPU cores per array/per trial
#SBATCH --mem=64G                 # memory
#SBATCH --time=08:00:00           # time limit (max)
#SBATCH --array=0-9              # run 10 trials (change as required)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dinelka@unc.edu  # change to your email

# Load required modules
module load python/3.9.6
module load cuda/11.3    # Load the correct CUDA version if needed
module load cudnn/8.2.1  # Optional: if your script needs it

# Run your script
python -u /work/users/d/i/dinelka/740_project/code/prediction/cnn.py