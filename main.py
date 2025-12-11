import argparse
import multiprocessing as mp
import subprocess
import sys

from compile_util import compiler_manager_task
from running_util import runner_manager_task

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
    parser.add_argument("--dump", type=str, default="dump/")
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("time", type=int, default=7200)
    
    args = parser.parse_args()

    num_gpus = get_gpu_count()
    print(f"[LOG] [MAIN] Detected {num_gpus} GPUs.", file=sys.stderr)

    # Create mp queues (autotuner -> compiler -> runner -> autotuner)
    autotuner_compile_queue = mp.Queue()
    compile_runner_queue = mp.Queue()
    runner_autotuner_queue = mp.Queue()

    # Instantiate compiler pool
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
        daemon=True,
        name="CompilerManager"
    )
    compiler_proc.start() 
    print(f"[LOG] [MAIN] Started Compiler Manager Process with {args.compile_workers} pool", file=sys.stderr)

    # Instantiate runner pool with num_gpus workers
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
        daemon=True,
        name="RunnerManager"
    )
    runner_proc.start()
    print(f"[LOG] [MAIN] Started Runner Manager Process with {num_gpus} pool", file=sys.stderr)
    
    # Instantiate autotuner
    pass

    # Start autotuning
    pass

    # Clean up pools after timeout
    pass
