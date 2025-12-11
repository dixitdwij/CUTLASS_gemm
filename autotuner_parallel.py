import os
import sys
import time
import multiprocessing as mp
from typing import List, Optional
from queue import Empty

from kernel_config import KernelConfig, SwizzlePolicy
from ncu_parser import parse_ncu_log

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
    WARP_DIVISORS = [(2, 2, 1), (4, 2, 1), (2, 4, 1), (1, 1, 1)]
    STAGES_LIST = [2, 3, 4, 5]
    SWIZZLE_FUNCS = [SwizzlePolicy.Identity, SwizzlePolicy.SplitK]
    SWIZZLE_N_VALUES = [1, 2, 4] 

    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, dim_m: int, dim_n: int, dim_k: int):
        self.input_queue = input_queue   # Queue to send configs to Compiler
        self.output_queue = output_queue # Queue to receive results from Runner
        self.best_config: Optional[KernelConfig] = None
        self.best_tflop: float = 0.0
        self.dim_m = dim_m
        self.dim_n = dim_n  
        self.dim_k = dim_k

    def generate_initial_configs(self) -> List[KernelConfig]:
        """Generates the search space via Cartesian product of valid parameters."""
        configs = []
        
        for inst_m, inst_n, inst_k in self.INST_SHAPES:
            for tb_m, tb_n, tb_k in self.TB_TILES:
                for w_div_m, w_div_n, w_div_k in self.WARP_DIVISORS:
                    
                    # Calculate Warp Shape based on TB and Divisor
                    w_m = tb_m // w_div_m
                    w_n = tb_n // w_div_n
                    w_k = tb_k // w_div_k

                    # Constraint: Warp size sanity check (avoid very small invalid warps)
                    if w_m < 16 or w_n < 8:
                        continue
                    
                    # Constraint: TB must be multiple of Warp
                    if tb_m % w_m != 0 or tb_n % w_n != 0 or tb_k % w_k != 0:
                        continue

                    for stages in self.STAGES_LIST:
                        for swizzle in self.SWIZZLE_FUNCS:
                            for swizzle_n in self.SWIZZLE_N_VALUES:
                                
                                # Create Config with stages explicitly passed
                                cfg = KernelConfig(
                                    TB_M=tb_m, TB_N=tb_n, TB_K=tb_k,
                                    W_M=w_m, W_N=w_n, W_K=w_k,
                                    INST_M=inst_m, INST_N=inst_n, INST_K=inst_k,
                                    stages=stages,
                                    swizzle_policy=swizzle,
                                    SwizzleN=swizzle_n
                                )
                                configs.append(cfg)

        print(f"[LOG] [AUTOTUNER] Generated {len(configs)} candidates.", file=sys.stderr)
        return configs

    def get_tflop_from_runtime(self, runtime_ms: float) -> float:
        if runtime_ms <= 0: return 0.0
        # 2 * M * N * K for Matrix Multiply FLOPs
        total_flops = 2.0 * self.dim_m * self.dim_n * self.dim_k
        # Convert ms to seconds (1e-3) and result to TFLOPs (1e-12)
        tflops = (total_flops / (runtime_ms / 1000.0)) / 1e12
        return tflops

    def tune(self, timeout_s: int):
        """
        Main execution loop.
        """
        candidates = self.generate_initial_configs()
        total_candidates = len(candidates)
        
        # Submit all jobs
        print(f"[LOG] [AUTOTUNER] Submitting {total_candidates} jobs to pipeline...", file=sys.stderr)
        for cfg in candidates:
            self.input_queue.put(cfg)
        
        print("[LOG] [AUTOTUNER] Jobs submitted. Listening for results...", file=sys.stderr)
        
        start_time = time.time()
        processed_count = 0

        while True:
            # Check for global timeout
            if time.time() - start_time > timeout_s:
                print(f"[LOG] [AUTOTUNER] Time limit of {timeout_s}s reached.", file=sys.stderr)
                break
            
            # Stop if all jobs are processed
            if processed_count >= total_candidates:
                print("[LOG] [AUTOTUNER] All candidates processed.", file=sys.stderr)
                break

            try:
                # Wait for result with short timeout to allow periodic global timeout checks
                result_config: KernelConfig = self.output_queue.get(timeout=5)
                processed_count += 1
                
                # Verify output file exists
                output_path = result_config.get_output_file_path()
                if not output_path or not os.path.exists(output_path):
                    # Could happen if compilation failed or execution crashed
                    continue

                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    parsed_data = parse_ncu_log(content)
                    
                    # Extract Duration from NCU parsed data
                    duration_ms = 0.0
                    if 'GPU Speed Of Light Throughput' in parsed_data:
                        duration_data = parsed_data['GPU Speed Of Light Throughput'].get('Duration', {})
                        duration_ms = float(duration_data.get('val', 0.0))
                    
                    if duration_ms > 0:
                        tflops = self.get_tflop_from_runtime(duration_ms)
                        
                        # Update Best
                        if tflops > self.best_tflop:
                            self.best_tflop = tflops
                            self.best_config = result_config
                            print(f"[SUCCESS] New Best: {tflops:.4f} TFLOPs | {result_config.kernel_id()}", file=sys.stderr)
                    else:
                        # Metric missing or zero duration usually indicates run failure or NCU parsing issue
                        pass

                except Exception as e:
                    print(f"[ERROR] [AUTOTUNER] Failed to parse results for {result_config.kernel_id()}: {e}", file=sys.stderr)

            except Empty:
                continue
        
        # Final Report
        print("\n" + "="*60)
        if self.best_config:
            print(f"AUTOTUNING COMPLETE.")
            print(f"Best Configuration Found:")
            print(f"  ID: {self.best_config.kernel_id()}")
            print(f"  Performance: {self.best_tflop:.4f} TFLOPs")
            print(f"  Stages: {self.best_config.stages}")
            print(f"  Swizzle: {self.best_config.swizzle_policy.name} (N={self.best_config.SwizzleN})")
        else:
            print("AUTOTUNING FAILED: No successful valid configurations found.")
        print("="*60 + "\n")