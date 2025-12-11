import multiprocessing as mp
from kernel_config import KernelConfig
import subprocess
import sys

timeout: int = 300
_dim_m: int
_dim_n: int
_dim_k: int
_dump_path: str = "dump/"


def _init_worker(dim_m: int, dim_n: int, dim_k: int, dump_path: str) -> None:
    global _dim_m
    global _dim_n
    global _dim_k
    global _dump_path
    _dim_m = dim_m
    _dim_n = dim_n
    _dim_k = dim_k
    _dump_path = dump_path


# Check if returning config every time leads to any issues with wrong entries in queue
def run_worker(config: KernelConfig) -> KernelConfig:
    binary_path = config.get_binary_path()
    output_file: str = _dump_path + config.kernel_id()

    # cmd = [f"./{binary_path}", str(_dim_m), str(_dim_n), str(_dim_k)]
    cmd = ['ncu', 
           '--launch-skip', '1',
           '--launch-count', '1',
           '--set', 'full',
           f"./{binary_path}", str(_dim_m), str(_dim_n), str(_dim_k)
    ]

    with open(output_file, 'w', encoding='utf-8') as outfile:
        result = subprocess.run(
            cmd, 
            stdout=outfile, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding='utf-8',
            timeout=timeout
        )

    if result.returncode == 0:
        print(f"[LOG] [RUNNER] Successfully ran kernel: {config.kernel_id()}", file=sys.stderr)
        config.register_output_file(output_file)  # Register the output file
        return config
    else:
        raise RuntimeError(f"[ERROR] [RUNNER] Execution failed for kernel: {config.kernel_id()}. Error: {result.stderr}") 


def runner_manager_task(
        input_queue: mp.Queue, 
        output_queue: mp.Queue, 
        num_run_workers: int, 
        dim_m: int, 
        dim_n: int, 
        dim_k: int, 
        dump_path: str
    ) -> None:
    global _dim_m
    global _dim_n
    global _dim_k
    global _dump_path
    _dim_m = dim_m
    _dim_n = dim_n
    _dim_k = dim_k
    _dump_path = dump_path

    print(f"[LOG] [RUNNER MANAGER] Starting runner manager with {num_run_workers} workers.", file=sys.stderr)
    print(f"[LOG] [RUNNER MANAGER] Registered Dimensions: M={_dim_m}, N={_dim_n}, K={_dim_k} Dumppath: {_dump_path}", file=sys.stderr)

    def result_callback(result: KernelConfig):
        print(f"[LOG] [RUNNER MANAGER] Runner worker completed for kernel: {result.kernel_id()}", file=sys.stderr)
        output_queue.put(result)

    def error_callback(e: BaseException):
        print(f"[ERROR] [RUNNER MANAGER] Runner worker encountered an error: {str(e)}", file=sys.stderr)

    with mp.Pool(processes=num_run_workers, initializer=_init_worker, initargs=(dim_m, dim_n, dim_k, dump_path)) as pool:
        while True:
            config: KernelConfig = input_queue.get()
            
            if config is None:
                print(f"[LOG] [RUNNER MANAGER] Received termination signal. Exiting.", file=sys.stderr)
                output_queue.put(None)  # Propagate termination signal
                break
        
            pool.apply_async(run_worker, args=(config,), callback=result_callback, error_callback=error_callback)

        pool.close()
        pool.join()

    print(f"[LOG] [RUNNER MANAGER] Runner Manager Terminating", file=sys.stderr)
    print(f"[LOG] [RUNNER MANAGER] Runner Manager Terminating", file=sys.stdout)
