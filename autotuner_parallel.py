import os
import sys
import time
import multiprocessing as mp
import random
from typing import List, Optional, Set
from queue import Empty

from kernel_config import KernelConfig, SwizzlePolicy
from ncu_parser import parse_ncu_log


class KernelPerformance:
    def __init__(self, parsed_data: dict):
        self.parsed_data = parsed_data
        self.duration_ms: float = parsed_data['GPU Speed Of Light Throughput']['Duration']['val']
        self.mem_throughput_pct: float = parsed_data['GPU Speed Of Light Throughput']['Memory Throughput']['val']
        self.sm__pct: float = parsed_data['GPU Speed Of Light Throughput']['Compute (SM) Throughput']['val']
        self.dram_throughput_pct: float = parsed_data['GPU Speed Of Light Throughput']['DRAM Throughput']['val']
        self.l1_throughput_pct: float = parsed_data['GPU Speed Of Light Throughput']['L1/TEX Cache Throughput']['val']
        self.l2_throughput_pct: float = parsed_data['GPU Speed Of Light Throughput']['L2 Cache Throughput']['val']
        self.ipc: float = parsed_data['Compute Workload Analysis']['Executed Ipc Active']['val']
        self.mem_max_bandwidth: float = parsed_data['Memory Workload Analysis']['Max Bandwidth']['val']
        self.l1_tex_hit_rate_pct: float = parsed_data['Memory Workload Analysis']['L1/TEX Hit Rate']['val']
        self.l2_hit_rate_pct: float = parsed_data['Memory Workload Analysis']['L2 Hit Rate']['val']
        self.reg_per_thread: int = parsed_data['Launch Statistics']['Registers Per Thread']['val']


class CutlassAutotunerParallel:
    # Tunable Parameters
    INST_SHAPES = [
        (16, 8, 8), 
        (16, 8, 16)
    ]
    TB_TILES = [
        (64, 64, 32),
        (64, 128, 32),
        (128, 64, 32),
        (128, 128, 32),
        (256, 128, 32),
        (128, 256, 32)
    ]
    # (Warp_M_Divisor, Warp_N_Divisor, Warp_K_Divisor)
    WARP_DIVISORS = [(2, 2, 1), (4, 2, 1), (2, 4, 1), (1, 1, 1)]
    STAGES_LIST = [2, 3, 4, 5]
    SWIZZLE_FUNCS = [SwizzlePolicy.Identity, SwizzlePolicy.SplitK]
    SWIZZLE_N_VALUES = [1, 2, 4] 

    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, dim_m: int, dim_n: int, dim_k: int, bar_size: int = 10):
        self.input_queue = input_queue   # Queue to send configs to Compiler
        self.output_queue = output_queue # Queue to receive results from Runner
        self.dim_m = dim_m
        self.dim_n = dim_n  
        self.dim_k = dim_k
        self.bar_size = bar_size
        
        self.best_config: Optional[KernelConfig] = None
        self.best_tflop: float = 0.0
        self.visited_configs: Set[str] = set()

    def get_random_config(self) -> KernelConfig:
        while True:
            # Randomly select parameters
            inst_m, inst_n, inst_k = random.choice(self.INST_SHAPES)
            tb_m, tb_n, tb_k = random.choice(self.TB_TILES)
            w_div_m, w_div_n, w_div_k = random.choice(self.WARP_DIVISORS)
            
            # Calculate Warp Shape
            w_m = tb_m // w_div_m
            w_n = tb_n // w_div_n
            w_k = tb_k // w_div_k

            # Validity Checks
            # Constraint: Warp size sanity check
            if w_m < 16 or w_n < 8:
                continue
            
            # Constraint: TB must be multiple of Warp
            if tb_m % w_m != 0 or tb_n % w_n != 0 or tb_k % w_k != 0:
                continue

            stages = random.choice(self.STAGES_LIST)
            swizzle = random.choice(self.SWIZZLE_FUNCS)
            swizzle_n = random.choice(self.SWIZZLE_N_VALUES)
            
            return KernelConfig(
                TB_M=tb_m, TB_N=tb_n, TB_K=tb_k,
                W_M=w_m, W_N=w_n, W_K=w_k,
                INST_M=inst_m, INST_N=inst_n, INST_K=inst_k,
                stages=stages,
                swizzle_policy=swizzle,
                SwizzleN=swizzle_n
            )

    def get_heuristic_config(self) -> Optional[KernelConfig]:
        """
        Placeholder for heuristic generation.
        Returns None to fall back to random generation.
        """
        # TODO: Implement heuristics based on self.best_config or recent results
        return None

    def get_tflop_from_runtime(self, runtime_ms: float) -> float:
        if runtime_ms <= 0: return 0.0
        total_flops = 2.0 * self.dim_m * self.dim_n * self.dim_k
        tflops = (total_flops / (runtime_ms / 1000.0)) / 1e12
        return tflops

    def tune(self, timeout_s: int):
        print(f"[LOG] [AUTOTUNER] Starting Bar Search with bar_size={self.bar_size}...", file=sys.stderr)
        
        start_time = time.time()
        pending_jobs = 0
        
        while True:
            current_time = time.time()
            if current_time - start_time > timeout_s:
                print(f"[LOG] [AUTOTUNER] Time limit of {timeout_s}s reached.", file=sys.stderr)
                break

            # Refill: Maintain Bar Size
            attempts = 0
            while pending_jobs < self.bar_size:
                # Escape infinite loop if search space is exhausted (or hard to find unique)
                if attempts > 200:
                    if pending_jobs == 0:
                        print("[LOG] [AUTOTUNER] Unable to generate new unique configs and no jobs pending. Stopping.", file=sys.stderr)
                        return
                    break 

                # Try Heuristics
                new_cfg = self.get_heuristic_config()
                
                # Fallback to Random
                if new_cfg is None:
                    new_cfg = self.get_random_config()

                # Uniqueness Check
                if new_cfg.kernel_id() not in self.visited_configs:
                    self.visited_configs.add(new_cfg.kernel_id())
                    self.input_queue.put(new_cfg)
                    pending_jobs += 1
                    attempts = 0 # Reset attempts on success
                else:
                    attempts += 1

            # Consume: Check for Results
            try:
                # Wait briefly for results to keep loop responsive to timeout
                result_config: KernelConfig = self.output_queue.get(timeout=1.0)
                pending_jobs -= 1
                
                # Verify output file exists
                output_path = result_config.get_output_file_path()
                if not output_path or not os.path.exists(output_path):
                    print(f"[WARN] [AUTOTUNER] No output found for {result_config.kernel_id()}", file=sys.stderr)
                    continue

                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    parsed_data = parse_ncu_log(content)
                    
                    # # Extract Duration
                    # duration_ms = 0.0
                    # if 'GPU Speed Of Light Throughput' in parsed_data:
                    #     duration_data = parsed_data['GPU Speed Of Light Throughput'].get('Duration', {})
                    #     # Handle case where val might be int or float
                    #     val = duration_data.get('val', 0.0)
                    #     duration_ms = float(val)

                    perf: KernelPerformance = KernelPerformance(parsed_data)
                    
                    if perf.duration_ms > 0:
                        tflops = self.get_tflop_from_runtime(perf.duration_ms)
                        
                        # Update Best
                        if tflops > self.best_tflop:
                            self.best_tflop = tflops
                            self.best_config = result_config
                            print(f"[SUCCESS] New Best: {tflops:.4f} TFLOPs | {result_config.kernel_id()}", file=sys.stderr)
                    else:
                        print(f"[WARN] [AUTOTUNER] Zero duration parsed for {result_config.kernel_id()}", file=sys.stderr)

                except Exception as e:
                    print(f"[ERROR] [AUTOTUNER] Failed to parse results for {result_config.kernel_id()}: {e}", file=sys.stderr)

            except Empty:
                # No results yet, continue loop to check timeout and refill
                continue
        
        print("\n" + "="*60)
        if self.best_config:
            print(f"AUTOTUNING COMPLETE.")
            print(f"Best Configuration Found:")
            print(f"  ID: {self.best_config.kernel_id()}")
            print(f"  Performance: {self.best_tflop:.4f} TFLOPs")
            print(f"  Stages: {self.best_config.stages}")
            print(f"  Swizzle: {self.best_config.swizzle_policy.name} (N={self.best_config.SwizzleN})")
            print(f"  Total Unique Configs Evaluated: {len(self.visited_configs)}")
        else:
            print("AUTOTUNING FAILED: No successful valid configurations found.")
        print("="*60 + "\n")