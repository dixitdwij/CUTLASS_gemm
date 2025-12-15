import os
import sys
import time
import multiprocessing as mp
import random
import copy
from typing import List, Optional, Set
from queue import Empty

from kernel_config import KernelConfig, SwizzlePolicy
from ncu_parser import parse_ncu_log

class KernelPerformance:
    def __init__(self, parsed_data: dict):
        self.parsed_data = parsed_data
        
        # Helper to safely extract values with defaults
        def get_val(section, metric, default=0.0):
            try:
                return parsed_data.get(section, {}).get(metric, {}).get('val', default)
            except:
                return default

        self.duration_ms: float = get_val('GPU Speed Of Light Throughput', 'Duration')
        self.mem_throughput_pct: float = get_val('GPU Speed Of Light Throughput', 'Memory Throughput')
        self.sm__pct: float = get_val('GPU Speed Of Light Throughput', 'Compute (SM) Throughput')
        self.dram_throughput_pct: float = get_val('GPU Speed Of Light Throughput', 'DRAM Throughput')
        self.l1_throughput_pct: float = get_val('GPU Speed Of Light Throughput', 'L1/TEX Cache Throughput')
        self.l2_throughput_pct: float = get_val('GPU Speed Of Light Throughput', 'L2 Cache Throughput')
        
        self.ipc: float = get_val('Compute Workload Analysis', 'Executed Ipc Active')
        
        self.mem_max_bandwidth: float = get_val('Memory Workload Analysis', 'Max Bandwidth')
        self.l1_tex_hit_rate_pct: float = get_val('Memory Workload Analysis', 'L1/TEX Hit Rate')
        self.l2_hit_rate_pct: float = get_val('Memory Workload Analysis', 'L2 Hit Rate')
        
        self.reg_per_thread: int = int(get_val('Launch Statistics', 'Registers Per Thread', 0))


class CutlassAutotunerParallel:
    # Tunable Parameters
    INST_SHAPES = [
        (16, 8, 8), 
        (16, 8, 16)
    ]
    # Sorted roughly by size (M*N) for heuristic stepping
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
        self.best_perf: Optional[KernelPerformance] = None # Store performance metrics of best config
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
            if self._is_valid(tb_m, tb_n, tb_k, w_m, w_n, w_k):
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

    def _is_valid(self, tb_m, tb_n, tb_k, w_m, w_n, w_k) -> bool:
        # Constraint: Warp size sanity check
        if w_m < 16 or w_n < 8:
            return False
        # Constraint: TB must be multiple of Warp
        if tb_m % w_m != 0 or tb_n % w_n != 0 or tb_k % w_k != 0:
            return False
        return True

    def get_heuristic_config(self) -> Optional[KernelConfig]:
        """
        Generates a neighbor configuration based on the performance characteristics 
        of the current best configuration.
        """
        if self.best_config is None or self.best_perf is None:
            return None
        
        # 1. Analyze Bottleneck
        perf = self.best_perf
        action = "explore" 

        # Heuristic Thresholds
        HIGH_REG_PRESSURE = 230 # Close to limit of 255
        HIGH_MEM_UTIL = 80.0
        HIGH_SM_UTIL = 80.0
        
        if perf.reg_per_thread > HIGH_REG_PRESSURE:
            # Bottleneck: Occupancy limited by registers
            action = "reduce_usage"
        elif perf.dram_throughput_pct > HIGH_MEM_UTIL or (perf.mem_throughput_pct > perf.sm__pct + 15.0):
            # Bottleneck: Memory Bandwidth
            action = "increase_reuse"
        elif perf.sm__pct > HIGH_SM_UTIL or (perf.sm__pct > perf.mem_throughput_pct + 15.0):
            # Bottleneck: Compute (or stuck on latency)
            action = "increase_work"
        else:
            action = "random_mutate"

        # 2. Mutate based on Action
        # Create a deep copy to modify
        base = self.best_config
        
        # Helper to get current TB index
        current_tb = (base.TB_M, base.TB_N, base.TB_K)
        try:
            tb_idx = self.TB_TILES.index(current_tb)
        except ValueError:
            tb_idx = 0

        # Attempt to generate a valid neighbor multiple times
        for _ in range(10): 
            new_tb_idx = tb_idx
            new_stages = base.stages
            
            mutation_choice = random.random()

            if action == "reduce_usage":
                # Strategy: Smaller Tiles OR Fewer Stages
                if mutation_choice < 0.6:
                    new_tb_idx = max(0, tb_idx - 1)
                else:
                    new_stages = max(min(self.STAGES_LIST), base.stages - 1)
            
            elif action == "increase_reuse":
                # Strategy: Larger Tiles OR More Stages (hide latency)
                if mutation_choice < 0.6:
                    new_tb_idx = min(len(self.TB_TILES) - 1, tb_idx + 1)
                else:
                    new_stages = min(max(self.STAGES_LIST), base.stages + 1)

            elif action == "increase_work":
                # Strategy: Larger Tiles (efficiency) or change Warp/Swizzle
                if mutation_choice < 0.5:
                    new_tb_idx = min(len(self.TB_TILES) - 1, tb_idx + 1)
                # Note: Swizzle/Warp changes handled implicitly by reconstruction logic below if we don't change TB/Stages
            
            # Fallback / Random pertubation (always a small chance)
            if random.random() < 0.2:
                if random.random() < 0.5:
                    new_stages = random.choice(self.STAGES_LIST)
                else:
                    new_tb_idx = random.randint(0, len(self.TB_TILES)-1)

            # Reconstruct Config
            tb_m, tb_n, tb_k = self.TB_TILES[new_tb_idx]
            
            # Keep warp divisor if possible, else pick random
            w_div_m, w_div_n, w_div_k = random.choice(self.WARP_DIVISORS)
            
            w_m = tb_m // w_div_m
            w_n = tb_n // w_div_n
            w_k = tb_k // w_div_k
            
            if self._is_valid(tb_m, tb_n, tb_k, w_m, w_n, w_k):
                # Swizzle mutation
                swizzle = base.swizzle_policy
                swizzle_n = base.SwizzleN
                if random.random() < 0.3:
                    swizzle = random.choice(self.SWIZZLE_FUNCS)
                    swizzle_n = random.choice(self.SWIZZLE_N_VALUES)

                return KernelConfig(
                    TB_M=tb_m, TB_N=tb_n, TB_K=tb_k,
                    W_M=w_m, W_N=w_n, W_K=w_k,
                    INST_M=base.INST_M, INST_N=base.INST_N, INST_K=base.INST_K, # Keep instruction shape same usually
                    stages=new_stages,
                    swizzle_policy=swizzle,
                    SwizzleN=swizzle_n
                )
        
        return None # Failed to find valid neighbor

    def get_tflop_from_runtime(self, runtime_ms: float) -> float:
        if runtime_ms <= 0: return 0.0
        total_flops = 2.0 * self.dim_m * self.dim_n * self.dim_k
        tflops = (total_flops / (runtime_ms / 1000.0)) / 1e12
        return tflops

    def tune(self, timeout_s: int):
        print(f"[LOG] [AUTOTUNER] Starting Bottleneck-Aware Search with bar_size={self.bar_size}...", file=sys.stderr)
        
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
                    attempts = 0 
                else:
                    attempts += 1

            # Consume: Check for Results
            try:
                result_config: KernelConfig = self.output_queue.get(timeout=1.0)
                pending_jobs -= 1
                
                output_path = result_config.get_output_file_path()
                if not output_path or not os.path.exists(output_path):
                    continue

                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    parsed_data = parse_ncu_log(content)
                    perf: KernelPerformance = KernelPerformance(parsed_data)
                    
                    if perf.duration_ms > 0:
                        tflops = self.get_tflop_from_runtime(perf.duration_ms)
                        
                        # Update Best
                        if tflops > self.best_tflop:
                            self.best_tflop = tflops
                            self.best_config = result_config
                            self.best_perf = perf # Store performance metrics for heuristics
                            print(f"[SUCCESS] New Best: {tflops:.4f} TFLOPs | {result_config.kernel_id()}", file=sys.stderr)
                            print(f"   [Reasons] SM: {perf.sm__pct:.1f}% | MEM: {perf.mem_throughput_pct:.1f}% | Regs: {perf.reg_per_thread}", file=sys.stderr)
                    else:
                        print(f"[WARN] [AUTOTUNER] Zero duration parsed for {result_config.kernel_id()}", file=sys.stderr)

                except Exception as e:
                    print(f"[ERROR] [AUTOTUNER] Failed to parse results for {result_config.kernel_id()}: {e}", file=sys.stderr)

            except Empty:
                continue
        
        print("\n" + "="*60)
        if self.best_config:
            print(f"AUTOTUNING COMPLETE.")
            print(f"Best Configuration Found:")
            print(f"  ID: {self.best_config.kernel_id()}")
            print(f"  Performance: {self.best_tflop:.4f} TFLOPs")
            print(f"  Stages: {self.best_config.stages}")
            print(f"  Swizzle: {self.best_config.swizzle_policy.name} (N={self.best_config.SwizzleN})")
            if self.best_perf:
                print(f"  Metrics: SM={self.best_perf.sm__pct}% MEM={self.best_perf.mem_throughput_pct}% Regs={self.best_perf.reg_per_thread}")
            print(f"  Total Unique Configs Evaluated: {len(self.visited_configs)}")
        else:
            print("AUTOTUNING FAILED: No successful valid configurations found.")
        print("="*60 + "\n")