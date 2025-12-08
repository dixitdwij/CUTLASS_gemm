import argparse
import subprocess
import sys


def get_gpu_count() -> int:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--list-gpus"], 
            encoding='utf-8'
        )
        num_gpus = len(output.strip().split('\n'))
        return num_gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f'[FATAL ERROR] [MAIN] nvidia-smi not found or failed to execute. Exiting.', file=sys.stderr)
        exit(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUTLASS Autotuner")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", type=str, choices=["BF_16", "FP_16", "FP_32"], required=True)
    parser.add_argument("--source", type=str, default="gemm_kernel.cu")
    parser.add_argument("--binary", type=str, default="bin/dyntuned_gemm")
    
    args = parser.parse_args()

    num_gpus = get_gpu_count()

    # Instantiate autotuner
    # tuner = CutlassAutotuner(source_file=args.source, binary_file=args.binary)
    # config = tuner.tune(args.m, args.n, args.k, dtype=args.dtype)

    # print(f"[AUTOTUNER] Optimal Configuration: {config}")
