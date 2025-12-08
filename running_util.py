import multiprocessing as mp
from kernel_config import KernelConfig
import subprocess
import sys


_dim_m: int
_dim_n: int
_dim_k: int
_dump_path: str = "dump/"

# Check if returning config every time leads to any issues with wrong entries in queue
def run_worker(config: KernelConfig) -> KernelConfig:
    binary_path = config.get_binary_path()
    output_file: str = _dump_path + config.kernel_id()

    cmd = [f"./{binary_path}", str(_dim_m), str(_dim_n), str(_dim_k)]
    try:
        with open(output_file, 'w') as outfile:
            result = subprocess.run(
                cmd, 
                stdout=outfile, 
                stderr=subprocess.PIPE, 
                text=True, 
                encoding='utf-8'
            )

        if result.returncode == 0:
            print(f"[LOG] [RUNNER] Successfully ran kernel: {config.kernel_id()}", file=sys.stderr)
            config.register_output_file(output_file)  # Register the output file
        else:
            print(f"[ERROR] [RUNNER] Kernel execution failed for {config.kernel_id()}.", file=sys.stderr)
            print(f"[ERROR] [RUNNER] {result.stderr}", file=sys.stderr)
        return config
    except Exception as e:
        print(f"[ERROR] [RUNNER] Execution failed for kernel: {config.kernel_id()}", file=sys.stderr)
        print(f"[ERROR] [RUNNER] Exception: {str(e)}", file=sys.stderr)
        return config
