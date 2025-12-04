import argparse
import os
import subprocess
import itertools
import re
import sys
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
import json # Used for config serialization

# --- Worker Function Definition ---

def worker_process(
    gpu_id: int, 
    task_queue: mp.Queue, 
    results_queue: mp.Queue, 
    source_file: str, 
    binary_base_path: str,
    dtype_flag: str,
    m: int, n: int, k: int
):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    while True:
        try:
            # Get a task from the queue. block=True waits until an item is available.
            # sentinel 'STOP' to stop worker
            task = task_queue.get() 
            
            if task == 'STOP':
                task_queue.put('STOP')  # For other workers before exiting
                print(f"[WORKER {gpu_id}] Stopping gracefully.", file=sys.stderr)
                break

            config_id, config = task
            
            # The binary path must be unique for each worker/config to avoid race conditions 
            # during compilation/execution. We use the config_id to make it unique.
            # Note: For an autotuner running many configs, a temp directory would be better,
            # but for simplicity, we use the unique binary path here.
            binary_path = f"{binary_base_path}_{config_id}" 

            # --- Compile Kernel ---
            
            cmd = ["nvcc", source_file, 
                   "-o", binary_path, 
                   "-O3", "-arch=sm_90", # Use sm_90 as in your original
                    "--expt-relaxed-constexpr",
                    "-std=c++17",
                    "-I./lib/cutlass/include",
                    "-I./lib/cutlass/tools/util/include",
                    "-DVERIFY"
            ]
            cmd.append(f"-D{dtype_flag}")
            
            # Add Tunable Parameters (as in the original implementation)
            cmd.append(f"-DTB_M={config['TB_M']}")
            cmd.append(f"-DTB_N={config['TB_N']}")
            cmd.append(f"-DTB_K={config['TB_K']}")
            cmd.append(f"-DW_M={config['W_M']}")
            cmd.append(f"-DW_N={config['W_N']}")
            cmd.append(f"-DW_K={config['W_K']}")
            cmd.append(f"-DINST_M={config['INST_M']}")
            cmd.append(f"-DINST_N={config['INST_N']}")
            cmd.append(f"-DINST_K={config['INST_K']}")

            cmd.append(f"-D{config['SwizzleStrategy']}")
            cmd.append(f"-DSwizzleN={config['SwizzleN']}")
            cmd.append(f"-DSTAGES={config['STAGES']}")

            perf = 0.0
            
            print(f"[WORKER {gpu_id}][ID:{config_id}] Attempting Compiling config: {config}", file=sys.stderr)
            try:
                # Suppress output unless error
                subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                print(f"[WORKER {gpu_id}][ID:{config_id}] Compilation success.", file=sys.stderr)

                # --- Run Kernel ---
                
                print(f"[WORKER {gpu_id}][ID:{config_id}] Attempting Running kernel with MNK=({m},{n},{k})", file=sys.stderr)
                # The binary is run by the worker process, which is already set to the correct GPU via CUDA_VISIBLE_DEVICES
                output: str = subprocess.check_output([f"./{binary_path}", str(m), str(n), str(k)], stderr=subprocess.STDOUT, encoding='utf-8')
                
                match_result = re.search(r"TFLOPs:\s*(\d+\.?\d*)", output)
                if match_result:
                    perf = float(match_result.group(1))
                    print(f"[WORKER {gpu_id}][ID:{config_id}] Kernel Execution succeeded perf: {perf} TFLOPs", file=sys.stderr)
                else: 
                    print(f"[WORKER {gpu_id}][ID:{config_id}] Kernel output format incompatible: No TFLOPs found", file=sys.stderr)
            
            except subprocess.CalledProcessError as e:
                # Error in either compilation or execution
                error_output = e.output.decode('utf-8').strip() if hasattr(e, 'output') else "Unknown error."
                stage = "Compilation" if "nvcc" in " ".join(cmd) else "Kernel Execution"
                print(f"[WORKER {gpu_id}][ID:{config_id}] {stage} failed: {error_output}", file=sys.stderr)
            except Exception as e:
                 print(f"[WORKER {gpu_id}][ID:{config_id}] An unexpected error occurred: {e}", file=sys.stderr)
            finally:
                # Clean up the temporary binary file after execution attempt
                try:
                    os.remove(binary_path)
                except OSError:
                    pass
                
            # Send result back to the main process
            results_queue.put({
                "config_id": config_id,
                "config": config,
                "perf": perf,
                "gpu_id": gpu_id
            })

        except mp.TimeoutError:
            # Should not happen with current logic, but good practice
            print(f"[WORKER {gpu_id}] Queue timeout, stopping.", file=sys.stderr)
            break
        except Exception as e:
            print(f"[WORKER {gpu_id}] Catastrophic error in worker: {e}", file=sys.stderr)
            break


class CutlassAutoTuner:
    # ... (Keep all class variables like INST_SHAPES, TB_TILES, etc., as in the original)
    INST_SHAPES = [
        (16, 8, 16),
        (16, 8, 8), 
        (1, 1, 1)
    ]
    TB_TILES = [
        (128, 128, 32),
        (128, 64, 32),
        (64, 128, 32),
        (64, 64, 32),
        (256, 128, 32),
        (128, 256, 32),
    ]
    WARP_DIVISORS = [(2, 2, 1), (4, 2, 1), (2, 4, 1), (1, 1, 1)]
    STAGES_LIST = [2, 3, 4, 5]
    SWIZZLE_FUNCS = ["SwizzleIdentity", "SwizzleKSplit"]
    SWIZZLE_N_VALUES = [1, 2, 4, 8]
    
    def __init__(self, source_file="gemm_kernel.cu", binary_base_path="bin/dyntuned_gemm"):
        self.source_file = source_file
        # Base path for the binary. Each worker will append a unique ID.
        self.binary_base_path = binary_base_path 
        self.results = []

    def _detect_gpus(self) -> int:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "-L"], 
                stderr=subprocess.STDOUT, 
                encoding='utf-8'
            )
            num_gpus = len([line for line in output.strip().split('\n') if line.startswith("GPU ")])
            print(f"[AUTOTUNER] Detected {num_gpus} GPUs on the node.", file=sys.stderr)
            return num_gpus if num_gpus > 0 else 1 # Fallback to 1 if no GPUs detected (e.g. for testing)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[AUTOTUNER] Could not run nvidia-smi. Assuming 1 GPU.", file=sys.stderr)
            return 1

    def generate_search_space(self) -> List[Tuple[int, Dict[str, int | str]]]:
        search_space_dicts = []
        search_space_dicts = self._manual_search_space() 
        # search_space_dicts = self._generate_search_space()

        # Assign unique ID to each config
        search_space_with_id = []
        for i, config in enumerate(search_space_dicts):
            search_space_with_id.append((i + 1, config)) # Start IDs from 1
        
        return search_space_with_id

    def _generate_search_space(self) -> List[Dict[str, int | str]]:
        search_space = []
        for (tb_m, tb_n, tb_k), (inst_m, inst_n, inst_k), stages, swizzle, swizzle_n in itertools.product(
            CutlassAutoTuner.TB_TILES,
            CutlassAutoTuner.INST_SHAPES,
            CutlassAutoTuner.STAGES_LIST,
            CutlassAutoTuner.SWIZZLE_FUNCS,
            CutlassAutoTuner.SWIZZLE_N_VALUES
        ):
            if tb_m < inst_m or tb_n < inst_n or tb_k < inst_k:
                continue

            for wm_div, wn_div, wk_div in [(1,1,1), (2,1,1), (1,2,1), (2,2,1), (4,1,1), (1,4,1)]:
                w_m = tb_m // wm_div
                w_n = tb_n // wn_div
                w_k = tb_k // wk_div
                
                if (w_m < inst_m or w_n < inst_n or w_k < inst_k):
                    continue
                
                if (tb_m % w_m != 0 or tb_n % w_n != 0 or tb_k % w_k != 0):
                    continue

                config = {
                    "TB_M": tb_m, "TB_N": tb_n, "TB_K": tb_k,
                    "W_M": w_m, "W_N": w_n, "W_K": w_k,
                    "INST_M": inst_m, "INST_N": inst_n, "INST_K": inst_k,
                    "STAGES": stages,
                    "SwizzleStrategy": swizzle, "SwizzleN": swizzle_n
                }
                search_space.append(config)
        return search_space

    def _manual_search_space(self) -> List[Dict[str, int | str]]:
        configs = [
            {"TB_M":128,"TB_N":256,"TB_K":64,"W_M":64,"W_N":64,"W_K":64,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":3, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":64,"TB_N":256,"TB_K":32,"W_M":32,"W_N":64,"W_K":32,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":4, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":128,"TB_N":128,"TB_K":32,"W_M":64,"W_N":64,"W_K":32,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":4, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":128,"TB_N":64,"TB_K":32,"W_M":64,"W_N":32,"W_K":32,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":4, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":64,"TB_N":128,"TB_K":32,"W_M":32,"W_N":64,"W_K":32,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":4, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":128,"TB_N":32,"TB_K":32,"W_M":64,"W_N":32,"W_K":32,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":4, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":64,"TB_N":32,"TB_K":32,"W_M":32,"W_N":32,"W_K":32,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":5, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":32,"TB_N":64,"TB_K":32,"W_M":32,"W_N":32,"W_K":32,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":5, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":128,"TB_N":128,"TB_K":64,"W_M":64,"W_N":64,"W_K":64,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":4, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":128,"TB_N":64,"TB_K":64,"W_M":64,"W_N":32,"W_K":64,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":4, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":64,"TB_N":128,"TB_K":64,"W_M":32,"W_N":64,"W_K":64,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":4, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":256,"TB_N":256,"TB_K":32,"W_M":64,"W_N":64,"W_K":32,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":3, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":256,"TB_N":128,"TB_K":32,"W_M":64,"W_N":64,"W_K":32,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":3, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":128,"TB_N":256,"TB_K":32,"W_M":64,"W_N":64,"W_K":32,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":3, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":64,"TB_N":64,"TB_K":32,"W_M":32,"W_N":32,"W_K":32,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":5, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":256,"TB_N":256,"TB_K":64,"W_M":64,"W_N":64,"W_K":64,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":3, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":256,"TB_N":128,"TB_K":64,"W_M":64,"W_N":64,"W_K":64,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":3, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":128,"TB_N":256,"TB_K":64,"W_M":64,"W_N":64,"W_K":64,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":4, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":256,"TB_N":256,"TB_K":64,"W_M":64,"W_N":64,"W_K":64,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":4, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
            {"TB_M":128,"TB_N":128,"TB_K":64,"W_M":64,"W_N":64,"W_K":64,"INST_M":16,"INST_N":8,"INST_K":16,"STAGES":3, "SwizzleStrategy":"SwizzleIdentity","SwizzleN":1,},
        ]
        return configs


    def tune(self, m: int, n: int, k: int, dtype: str = "BF_16") -> Dict[str, int | str]:
        
        dtype_flag = {
            "FP_32": "FP_32",
            "FP_16": "FP_16",
            "BF_16": "BF_16"
        }.get(dtype, "BF_16")
        
        print(f"[AUTOTUNER] Starting Parallel Autotuning for {dtype} MNK=({m},{n},{k})...")

        # 1. Setup
        num_gpus = self._detect_gpus()
        search_space = self.generate_search_space()
        
        print(f"[AUTOTUNER] Generated {len(search_space)} potential configurations to test on {num_gpus} GPUs.")

        if not search_space:
            print("[AUTOTUNER] Search space is empty.", file=sys.stderr)
            return {}

        # Use Manager to create objects that can be shared between processes
        # This is strictly not needed for Queue/Process, but is good practice.
        manager = mp.Manager()
        task_queue = manager.Queue()
        results_queue = manager.Queue()
        
        # Populate the task queue
        for config_id, config in search_space:
            task_queue.put((config_id, config))
            
        # Add 'STOP' sentinel for each worker
        for _ in range(num_gpus):
            task_queue.put('STOP')

        # 2. Start Worker Processes
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=worker_process, 
                args=(
                    gpu_id, 
                    task_queue, 
                    results_queue, 
                    self.source_file, 
                    self.binary_base_path,
                    dtype_flag,
                    m, n, k
                )
            )
            processes.append(p)
            p.start()

        # 3. Collect Results
        best_perf = 0.0
        best_config = None
        best_config_id = None
        
        # Wait for all search space items to be processed
        total_tasks = len(search_space)
        processed_tasks = 0

        while processed_tasks < total_tasks:
            result = results_queue.get()
            processed_tasks += 1
            
            perf = result["perf"]
            config = result["config"]
            config_id = result["config_id"]
            
            print(f"[AUTOTUNER] Processed {processed_tasks}/{total_tasks} tasks. Last: ID:{config_id}, Perf: {perf} TFLOPs", file=sys.stderr)
            
            if perf > best_perf:
                best_perf = perf
                best_config = config
                best_config_id = config_id

        # 4. Cleanup
        # Wait for all workers to terminate (they should have all received 'STOP')
        for p in processes:
            p.join()
            
        print(f"[AUTOTUNER] Autotuning completed.")
        if best_config:
            print(f"[AUTOTUNER] Best Performance: {best_perf} TFLOPs with config ID: {best_config_id} and config: {best_config}")
            # Add the ID to the returned config for complete logging
            best_config["CONFIG_ID"] = best_config_id
            return best_config
        else:
            print("[AUTOTUNER] No valid configuration found.", file=sys.stderr)
            return {}

if __name__ == "__main__":
    # Ensure bin directory exists for the binaries
    os.makedirs("bin", exist_ok=True) 

    parser = argparse.ArgumentParser(description="CUTLASS Parallel Autotuner")
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--dtype", type=str, choices=["BF_16", "FP_16", "FP_32"], default="BF_16")
    # Source file name (e.g., dyntuned_gemm.cu)
    parser.add_argument("--source", type=str, default="dyntuned_gemm.cu") 
    # Base binary name. Workers will append a unique ID.
    parser.add_argument("--binary", type=str, default="bin/dyntuned_gemm") 
    
    args = parser.parse_args()

    # The multiprocessing module needs to know what to run on Windows,
    # so we wrap the main logic in an explicit check.
    # On Linux/macOS, it's often run implicitly.
    if sys.platform.startswith('win'):
        mp.freeze_support()

    tuner = CutlassAutoTuner(source_file=args.source, binary_base_path=args.binary)
    config = tuner.tune(args.m, args.n, args.k, dtype=args.dtype)

    print("-" * 50)
    print(f"Optimal Configuration Found (JSON Output):")
    print(json.dumps(config, indent=4))