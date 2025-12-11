import os
import subprocess
import itertools
import re
import sys
from typing import List, Dict, Tuple, Optional
from ncu_parser import parse_ncu_log
from kernel_config import KernelConfig
import multiprocessing as mp

class CutlassAutotunerParallel:
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

    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, dim_m: int, dim_n: int, dim_k: int):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.best_config: Optional[KernelConfig] = None
        self.best_tflop: Optional[float] = None
        self.dim_m = dim_m
        self.dim_n = dim_n  
        self.dim_k = dim_k

    def generate_initial_configs(self) -> List[KernelConfig]:
        # Generate initial kernels to start the search
        pass

    def generate_next_candidate(self, current_config: KernelConfig, profile output) -> Optional[KernelConfig]:
        # Given 
        pass

    def get_tflop_from_runtime(self, runtime_ms: float) -> float:
        total_flops = 2 * self.dim_m * self.dim_n * self.dim_k
        tflops = (total_flops / runtime_ms) / 1e6  
        return tflops
    

# MP script to add initial kernels to the queue, listen for results, compute next candidates, and add them to the queue
pass


    