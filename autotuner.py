import argparse
import os
import subprocess
import itertools
import re
import sys
from typing import List, Dict, Tuple, Optional


class CutlassAutoTuner:
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

    def __init__(self, source_file="dyntuned_gemm.cu", binary_path="bin/dyntuned_gemm"):
        self.source_file = source_file
        self.binary_path = binary_path
        self.results = []

    def compile_kernel(self, config: Dict, dtype_flag: str) -> bool:
        cmd = ["nvcc", self.source_file, 
               "-o", self.binary_path, 
               "-O3", "-arch=sm_90",
                "--expt-relaxed-constexpr",
                "-std=c++17",
                "-I./lib/cutlass/include",
                "-I./lib/cutlass/tools/util/include",
                "-DVERIFY"
        ]
        cmd.append(f"-D{dtype_flag}")
        
        # Add Tunable Parameters
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

        print(f"[AUTOTUNER] Attempting Compiling with config: {config}", file=sys.stderr)
        try:
            # Suppress output unless error
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            print(f"[AUTOTUNER] Compilation success for config: {config}", file=sys.stderr)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[AUTOTUNER] Compilation failed for config: {config}", file=sys.stderr)
            print(e, sys.stderr)
            return False
        
    def run_kernel(self, m: int, n: int, k: int) -> float:
        try:
            print(f"[AUTOTUNER] Attempting Running kernel with MNK=({m},{n},{k})", file=sys.stderr)
            output: str = subprocess.check_output([f"./{self.binary_path}", str(m), str(n), str(k)], stderr=subprocess.STDOUT, encoding='utf-8')
            print(f"[AUTOTUNER] Kernel Execution Output:\n{output}", file=sys.stderr)
            match_result = re.search(r"TFLOPs:\s*(\d+\.?\d*)", output)
            if match_result:
                perf = match_result.group(1)
                print(f"[AUTOTUNER] Kernel Execution succeeded perf: {perf}", file=sys.stderr)
                return float(perf)
            else: 
                print(f"[AUTOTUNER] Kernel output format incompatible: No TFLOPs found", file=sys.stderr)
                return 0.0
            
        except subprocess.CalledProcessError as e:
            print(f"[AUTOTUNER] Kernel Execution failed: {e.output.decode('utf-8')}", file=sys.stderr)
            return 0.0
        
    def generate_search_space(self) -> List[Dict[str, int | str]]:
        search_space = []
        for (tb_m, tb_n, tb_k), (inst_m, inst_n, inst_k), stages, swizzle, swizzle_n in itertools.product(
            CutlassAutoTuner.TB_TILES,
            CutlassAutoTuner.INST_SHAPES,
            CutlassAutoTuner.STAGES_LIST,
            CutlassAutoTuner.SWIZZLE_FUNCS,
            CutlassAutoTuner.SWIZZLE_N_VALUES
        ):
            if tb_m < inst_m or tb_n < inst_n or tb_k < inst_k:
                continue    # TB must be greater than INST :)

            for wm_div, wn_div, wk_div in [(1,1,1), (2,1,1), (1,2,1), (2,2,1), (4,1,1), (1,4,1)]:
                w_m = tb_m // wm_div
                w_n = tb_n // wn_div
                w_k = tb_k // wk_div
                
                if (w_m < inst_m or w_n < inst_n or w_k < inst_k):
                    continue    # Warp size must be valid and >= Instruction
                
                if (tb_m % w_m != 0 or tb_n % w_n != 0 or tb_k % w_k != 0):
                    continue    # Warp size must be integer result of division

                config = {
                    "TB_M": tb_m,
                    "TB_N": tb_n,
                    "TB_K": tb_k,
                    "W_M": w_m,
                    "W_N": w_n,
                    "W_K": w_k,
                    "INST_M": inst_m,
                    "INST_N": inst_n,
                    "INST_K": inst_k,
                    "STAGES": stages,
                    "SwizzleStrategy": swizzle,
                    "SwizzleN": swizzle_n
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

        print(f"[AUTOTUNER] Starting Autotuning for {dtype} MNK=({m},{n},{k})...")

        # search_space = self.generate_search_space()
        search_space = self._manual_search_space()
        print(f"[AUTOTUNER] Generated {len(search_space)} potential configurations.")

        best_perf = 0.0
        best_config = None

        for config in search_space:
            if not self.compile_kernel(config, dtype_flag):
                continue    # INvalid tuner params
            
            perf = self.run_kernel(m, n, k)
            if perf > best_perf:
                best_perf = perf
                best_config = config

        print(f"[AUTOTUNER] Autotuning completed.")
        if best_config:
            print(f"[AUTOTUNER] Best Performance: {best_perf} TFLOPs with config: {best_config}")
            return best_config
        else:
            print("[AUTOTUNER] No valid configuration found.")
            return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUTLASS Autotuner")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", type=str, choices=["BF_16", "FP_16", "FP_32"], required=True)
    parser.add_argument("--source", type=str, default="gemm_kernel.cu")
    parser.add_argument("--binary", type=str, default="bin/dyntuned_gemm")
    
    args = parser.parse_args()

    tuner = CutlassAutoTuner(source_file=args.source)
    config = tuner.tune(args.m, args.n, args.k, dtype=args.dtype)

    print(f"[AUTOTUNER] Optimal Configuration: {config}")