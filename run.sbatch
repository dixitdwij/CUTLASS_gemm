#!/bin/bash

#SBATCH --job-name=dphpc-gemm
#SBATCH --time=5:00
#SBATCH --account=dphpc
#SBATCH --gpus=5060ti:1
#SBATCH --output=/home/ddixit/cutlass_baseline/output/dphpc-gemm_%j.out
#SBATCH --error=/home/ddixit/cutlass_baseline/output/dphpc-gemm_%j.err

python3 autotuner.py --m 8192 --n 8192 --k 2048 --dtype=BF_16 --source dyntuned_gemm.cu --binary bin/dyntuned_gemm

echo "[SBATCH] Run completed"