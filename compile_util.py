import multiprocessing as mp
from kernel_config import KernelConfig
import subprocess
import sys


# Glabal variables (check setup)
_source_file: str = "dyntuned_gemm.cu"
_bin_path: str = "bin/"
_dtype_str: str|None = None


def _init_worker(source_file: str, bin_path: str, dtype_str: str) -> None:
    # TODO: Cehck if thisis reqd as global vars are set by the manager before spawning pool
    global _source_file
    global _bin_path
    global _dtype_str
    _source_file = source_file
    _bin_path = bin_path
    _dtype_str = dtype_str


def compilation_worker(config: KernelConfig) -> KernelConfig:
    binary_file = _bin_path + config.kernel_id()
    cmd = ["nvcc", _source_file, 
            "-o", binary_file, 
            "-O3", "-arch=sm_90",
            "--expt-relaxed-constexpr",
            "-std=c++17",
            "-I./lib/cutlass/include",
            "-I./lib/cutlass/tools/util/include",
            f'-D{_dtype_str}',
            *config.compilation_flags()
    ]
    
    try:
        # result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,encoding='utf-8')
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        print(f"[LOG] [COMPILER] Successfully compiled kernel: {config.kernel_id()}", file=sys.stderr)
        config.register_compilation(binary_file, True)
        return config
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] [COMPILER] Compilation failed for kernel: {config.kernel_id()}", file=sys.stderr)
        print(f"[ERROR] [COMPILER] {e.stderr}")   # use e.output.decode("utf-8") if isinstance(e.output, bytes) else e.output for full error log
        config.register_compilation("", False)
        raise e
    

def compiler_manager_task(
        input_queue: mp.Queue, 
        output_queue: mp.Queue, 
        num_compile_workers: int, 
        source_file: str, 
        bin_path: str, 
        dtype_str: str
    ) -> None:
    global _source_file
    global _bin_path
    global _dtype_str
    _source_file = source_file
    _bin_path = bin_path
    _dtype_str = dtype_str

    print(f"[LOG] [Compiler Manager] Registered source_file:{_source_file} bin_path:{_bin_path} datatype:{_dtype_str}", file=sys.stderr)
    print(f"[LOG] [Compiler Manager] Starting with pool size: {num_compile_workers}", file=sys.stderr)

    def result_callback(result: KernelConfig):
        print(f"[LOG] [Compiler Manager] Compilation worker successful, scheduled for running", file=sys.stderr)
        output_queue.put(result)

    def error_callback(error: BaseException):
        # error type annotated as BaseException to make linter happy
        # subprocess.CalledProcessError will only be passed here (not that it matters much)
        print(f"[ERROR] [Compiler Manager] Compilation worker encountered an error: {error}", file=sys.stderr)

    with mp.Pool(processes=num_compile_workers, initializer=_init_worker, initargs=(source_file, bin_path, dtype_str)) as pool:
        while True:
            config: KernelConfig = input_queue.get()
            
            if config is None:
                print(f"[LOG] [Compiler Manager] Received termination signal. Exiting.", file=sys.stderr)
                output_queue.put(None)  # Propagate termination signal
                break
        
            pool.apply_async(compilation_worker, args=(config,), callback=result_callback, error_callback=error_callback)

        pool.close()
        pool.join()
    
    print(f"[LOG] [Compiler Manager] Compilation Manager Terminating", file=sys.stderr)
    print(f"[LOG] [Compiler Manager] Compilation Manager Terminating", file=sys.stdout)
