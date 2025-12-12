#!/bin/bash

#SBATCH --job-name=dphpc-gemm
#SBATCH --time=20:00
#SBATCH --account=dphpc
#SBATCH --gpus=5060ti:1
#SBATCH --output=/home/ddixit/CUTLASS_autotuner/output/dphpc-gemm_%j.out
#SBATCH --error=/home/ddixit/CUTLASS_autotuner/output/dphpc-gemm_%j.err

# source .venv/bin/activate
# nvcc -O3 -std=c++17 \
#     -I/usr/include/python3.12 \
#     -I/home/ddixit/cutlass_baseline/.venv/lib/python3.12/site-packages/pybind11/include \
#     -Xcompiler -fPIC \
#     --shared cublass_gemm.cu \
#     -lcublas -lcudart \
#     -o cublass_gemm$(python3-config --extension-suffix)
echo "[SBATCH] Cublas baseline set up"

python3 main.py --m 8192 --n 8192 --k 256 --dtype=BF_16 --source dyntuned_gemm.cu --binary bin/dyntuned_gemm --dump dump/ --compile-workers 2 --time 720

echo "[SBATCH] Run completed"