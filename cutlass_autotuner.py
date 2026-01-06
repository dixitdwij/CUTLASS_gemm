import os
import sys
import time
import multiprocessing as mp
import random
import copy
from typing import List, Optional, Set
from queue import Empty

from kernel_config import KernelConfig, SwizzlePolicy
from ncu_parser import parse_ncu_log, KernelPerformance
from autotuner_policy import get_default_policy, ActionType

class CutlassAutotunerParallel:
    # 1. STRUCTURED SEARCH SPACE
    
    # Fixed Instruction Shapes (Usually fixed by hardware/precision, but can be searched)
    INST_SHAPES = [
        (16, 8, 8), 
        (16, 8, 16)
    ]

    # Ordered by Size (M*N*K) then Aspect Ratio
    # Index 0 is smallest tile, Index -1 is largest
    TB_TILES = [
        (64, 64, 32),    # Size: 131k
        (64, 128, 32),   # Size: 262k
        (128, 64, 32),   # Size: 262k
        (128, 128, 32),  # Size: 524k
        (128, 256, 32),  # Size: 1M
        (256, 128, 32)   # Size: 1M
    ]

    # Ordered by Total Warps (Div_M * Div_N * Div_K)
    # Index 0 is least parallelism (1 warp), Index -1 is max parallelism
    WARP_DIVISORS = [
        (1, 1, 1),       # 1 Warp
        (2, 1, 1),       # 2 Warps 
        (1, 2, 1),       # 2 Warps
        (2, 2, 1),       # 4 Warps
        (4, 2, 1),       # 8 Warps
        (2, 4, 1)        # 8 Warps
    ]

    STAGES_LIST = [2, 3, 4, 5]
    
    # For Swizzle, we treat Identity < SplitK as a "mode switch"
    # and SwizzleN as an intensity parameter
    SWIZZLE_FUNCS = [SwizzlePolicy.Identity, SwizzlePolicy.SplitK]
    SWIZZLE_N_VALUES = [1, 2, 4] 

    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, dim_m: int, dim_n: int, dim_k: int, bar_size: int = 10):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.dim_m = dim_m
        self.dim_n = dim_n  
        self.dim_k = dim_k
        self.bar_size = bar_size
        
        self.best_config: Optional[KernelConfig] = None
        self.best_perf: Optional[KernelPerformance] = None 
        self.best_tflop: float = 0.0
        self.visited_configs: Set[str] = set()

        # Initialize the Composible Policy Engine
        self.policy_engine = get_default_policy()

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

    def get_config_indices(self, config: KernelConfig) -> dict:
        """Helper to recover the indices of a given config in the search lists."""
        indices = {}
        
        # Tile Index
        try:
            indices['TB_TILE'] = self.TB_TILES.index((config.TB_M, config.TB_N, config.TB_K))
        except ValueError:
            indices['TB_TILE'] = 0

        # Warp Divisor Index
        # We have to reverse calculate divisors because config stores absolute W_M
        div_m, div_n, div_k = config.TB_M // config.W_M, config.TB_N // config.W_N, config.TB_K // config.W_K
        try:
            indices['WARP_DIV'] = self.WARP_DIVISORS.index((div_m, div_n, div_k))
        except ValueError:
            indices['WARP_DIV'] = 0

        # Stages
        try:
            indices['STAGES'] = self.STAGES_LIST.index(config.stages)
        except ValueError:
            indices['STAGES'] = 0
            
        # Swizzle N
        try:
            indices['SWIZZLE_N'] = self.SWIZZLE_N_VALUES.index(config.SwizzleN)
        except ValueError:
            indices['SWIZZLE_N'] = 0

        return indices

    def get_neighbor_config(self, base: KernelConfig, param: str, direction: int) -> Optional[KernelConfig]:
        """
        Structural mutation: Moves adjacent in the search lists.
        param: matches ActionType (TB_TILE, WARP_DIV, STAGES, SWIZZLE_N, SWIZZLE_POLICY, RANDOM_WARP)
        direction: +1 (Increase), -1 (Decrease), or 0 (Special/Toggle)
        """
        indices = self.get_config_indices(base)
        
        # Start with copy of parameters
        tb_m, tb_n, tb_k = base.TB_M, base.TB_N, base.TB_K
        w_m, w_n, w_k = base.W_M, base.W_N, base.W_K
        stages = base.stages
        swizzle = base.swizzle_policy
        swizzle_n = base.SwizzleN

        # --- Handle Params ---
        
        if param == ActionType.TB_TILE:
            current_idx = indices['TB_TILE']
            new_idx = current_idx + direction
            if 0 <= new_idx < len(self.TB_TILES):
                tb_m, tb_n, tb_k = self.TB_TILES[new_idx]
                # Recalculate Warp using old divisors
                old_div_idx = indices['WARP_DIV']
                d_m, d_n, d_k = self.WARP_DIVISORS[old_div_idx]
                w_m, w_n, w_k = tb_m // d_m, tb_n // d_n, tb_k // d_k
            else:
                return None

        elif param == ActionType.STAGES:
            current_idx = indices['STAGES']
            new_idx = current_idx + direction
            if 0 <= new_idx < len(self.STAGES_LIST):
                stages = self.STAGES_LIST[new_idx]
            else:
                return None

        elif param == ActionType.WARP_DIV:
            current_idx = indices['WARP_DIV']
            new_idx = current_idx + direction
            if 0 <= new_idx < len(self.WARP_DIVISORS):
                d_m, d_n, d_k = self.WARP_DIVISORS[new_idx]
                w_m, w_n, w_k = tb_m // d_m, tb_n // d_n, tb_k // d_k
            else:
                return None

        elif param == ActionType.SWIZZLE_N:
            current_idx = indices['SWIZZLE_N']
            new_idx = current_idx + direction
            if 0 <= new_idx < len(self.SWIZZLE_N_VALUES):
                swizzle_n = self.SWIZZLE_N_VALUES[new_idx]
            else:
                return None

        elif param == ActionType.SWIZZLE_POLICY:
            # Toggle between Identity and SplitK
            if swizzle == SwizzlePolicy.Identity:
                swizzle = SwizzlePolicy.SplitK
            else:
                swizzle = SwizzlePolicy.Identity

        elif param == ActionType.RANDOM_WARP:
            # Pick a random warp divisor different from current
            current_idx = indices['WARP_DIV']
            choices = [i for i in range(len(self.WARP_DIVISORS)) if i != current_idx]
            if not choices: return None
            new_idx = random.choice(choices)
            d_m, d_n, d_k = self.WARP_DIVISORS[new_idx]
            w_m, w_n, w_k = tb_m // d_m, tb_n // d_n, tb_k // d_k

        else:
            return None

        # --- Validate ---
        if not self._is_valid(tb_m, tb_n, tb_k, w_m, w_n, w_k):
            return None

        return KernelConfig(
            TB_M=tb_m, TB_N=tb_n, TB_K=tb_k,
            W_M=w_m, W_N=w_n, W_K=w_k,
            INST_M=base.INST_M, INST_N=base.INST_N, INST_K=base.INST_K,
            stages=stages,
            swizzle_policy=swizzle,
            SwizzleN=swizzle_n
        )

    def get_heuristic_config(self) -> Optional[KernelConfig]:
        """
        Uses the Policy Engine to suggest the next configuration.
        """
        if self.best_config is None or self.best_perf is None:
            return None
        
        # 1. Ask Policy Engine for an Action
        action = self.policy_engine.evaluate(self.best_perf)
        
        if action:
            # 2. Try to apply the action
            print(f"[LOG] [AUTOTUNER] Policy Triggered: {action}", file=sys.stderr)
            neighbor = self.get_neighbor_config(self.best_config, action.param, action.direction)
            if neighbor:
                return neighbor
            else:
                print(f"[LOG] [AUTOTUNER] Policy action failed (boundary or invalid). Fallback to random.", file=sys.stderr)
        
        # 3. Fallback: Random Mutation if no policy triggered or action failed
        # Just pick a random dimension to perturb
        dims = [ActionType.TB_TILE, ActionType.WARP_DIV, ActionType.STAGES, ActionType.SWIZZLE_N]
        random_dim = random.choice(dims)
        random_dir = random.choice([-1, 1])
        
        return self.get_neighbor_config(self.best_config, random_dim, random_dir)

    def get_tflop_from_runtime(self, runtime_ms: float) -> float:
        if runtime_ms <= 0: return 0.0
        total_flops = 2.0 * self.dim_m * self.dim_n * self.dim_k
        tflops = (total_flops / (runtime_ms / 1000.0)) / 1e12
        return tflops

    def tune(self, timeout_s: int):
        print(f"[LOG] [AUTOTUNER] Starting Policy-Driven Search with bar_size={self.bar_size}...", file=sys.stderr)
        
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
