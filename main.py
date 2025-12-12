import argparse
import multiprocessing as mp
import subprocess
import sys
import os
import time

from compile_util import compiler_manager_task
from running_util import runner_manager_task
from autotuner_parallel import CutlassAutotunerParallel

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
        sys.exit(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUTLASS Autotuner")
    parser.add_argument("--m", type=int, required=True, help="Matrix dimension M")
    parser.add_argument("--n", type=int, required=True, help="Matrix dimension N")
    parser.add_argument("--k", type=int, required=True, help="Matrix dimension K")
    parser.add_argument("--dtype", type=str, choices=["BF_16", "FP_16", "FP_32"], required=True, help="Data type")
    parser.add_argument("--source", type=str, default="dyntuned_gemm.cu", help="Source CUDA file")
    parser.add_argument("--binary", type=str, default="bin/dyntuned_gemm", help="Path prefix for generated binaries")
    parser.add_argument("--dump", type=str, default="dump/", help="Directory to dump NCU logs")
    parser.add_argument("--compile-workers", type=int, default=4, help="Number of concurrent compilation processes")
    parser.add_argument("--time", type=int, default=7200, help="Autotuning time budget in seconds")
    parser.add_argument("--bar-size", type=int, default=10, help="Target number of candidates active in the pipeline (Bar Search size)")
    
    args = parser.parse_args()

    # Ensure output directories exist
    if not os.path.exists(args.dump):
        os.makedirs(args.dump, exist_ok=True)
    
    bin_dir = os.path.dirname(args.binary)
    if bin_dir and not os.path.exists(bin_dir):
        os.makedirs(bin_dir, exist_ok=True)

    num_gpus = get_gpu_count()
    print(f"[LOG] [MAIN] Detected {num_gpus} GPUs.", file=sys.stderr)

    autotuner_compile_queue = mp.Queue()
    compile_runner_queue = mp.Queue()
    runner_autotuner_queue = mp.Queue()

    # Compiler Manager
    compiler_proc: mp.Process = mp.Process(
        target=compiler_manager_task,
        args=(
            autotuner_compile_queue,
            compile_runner_queue,
            args.compile_workers,
            args.source,
            args.binary,
            args.dtype
        ),
        daemon=False, 
        name="CompilerManager"
    )
    compiler_proc.start() 
    print(f"[LOG] [MAIN] Started Compiler Manager with {args.compile_workers} workers", file=sys.stderr)

    # Runner Manager 
    runner_proc: mp.Process = mp.Process(
        target=runner_manager_task,
        args=(
            compile_runner_queue,
            runner_autotuner_queue,
            num_gpus,
            args.m,
            args.n,
            args.k,
            args.dump
        ),
        daemon=False,
        name="RunnerManager"
    )
    runner_proc.start()
    print(f"[LOG] [MAIN] Started Runner Manager with {num_gpus} workers", file=sys.stderr)
    
    # Autotuner Logic
    autotuner = CutlassAutotunerParallel(
        input_queue=autotuner_compile_queue,
        output_queue=runner_autotuner_queue,
        dim_m=args.m,
        dim_n=args.n,
        dim_k=args.k,
        bar_size=args.bar_size
    )

    # Start Tuning
    print(f"[LOG] [MAIN] Starting autotuning loop. Budget: {args.time}s, Bar Size: {args.bar_size}", file=sys.stderr)
    try:
        autotuner.tune(timeout_s=args.time)
    except KeyboardInterrupt:
        print("\n[LOG] [MAIN] Interrupted by user.", file=sys.stderr)
    except Exception as e:
        print(f"\n[ERROR] [MAIN] Critical exception: {e}", file=sys.stderr)

    # Cleanup
    print("[LOG] [MAIN] Cleaning up processes...", file=sys.stderr)

    # 1. Signal Compiler Manager to exit
    # The compiler manager logic breaks its loop when it receives None
    autotuner_compile_queue.put(None)
    
    # Wait for compiler to drain and close
    compiler_proc.join(timeout=10)
    if compiler_proc.is_alive():
        print("[LOG] [MAIN] Compiler Manager did not exit gracefully, terminating...", file=sys.stderr)
        compiler_proc.terminate()

    # 2. Signal Runner Manager to exit
    # Now that compiler is closed, we can safely send the stop signal to the runner
    # (usually runner gets input from compiler, so we inject the None into that queue)
    compile_runner_queue.put(None)
    
    runner_proc.join(timeout=10)
    if runner_proc.is_alive():
        print("[LOG] [MAIN] Runner Manager did not exit gracefully, terminating...", file=sys.stderr)
        runner_proc.terminate()

    print("[LOG] [MAIN] Exiting.", file=sys.stderr)