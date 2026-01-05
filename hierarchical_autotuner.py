from cutlass_autotuner import CutlassAutotunerParallel, KernelConfig, SwizzlePolicy, KernelPerformance
from ncu_parser import parse_ncu_log
import os 
import sys
import time


class HierarchicalAutotuner(CutlassAutotunerParallel):
    def tune(self, timeout_s: int):
        print(f"[LOG] [AUTOTUNER] Starting Hierarchical Search...", file=sys.stderr)
        
        # --- PHASE 1: Coarse Search (Find Best Tile) ---
        print(f"[LOG] [AUTOTUNER] Phase 1: Testing all Tile Shapes...", file=sys.stderr)
        
        # Generate candidates: All Tiles, Fixed Stages=3, Fixed Swizzle=Identity
        phase_1_configs = []
        for tb in self.TB_TILES:
            # Default Warp calc
            w_m, w_n, w_k = tb[0]//2, tb[1]//2, tb[2] 
            
            cfg = KernelConfig(
                TB_M=tb[0], TB_N=tb[1], TB_K=tb[2],
                W_M=w_m, W_N=w_n, W_K=w_k,
                INST_M=16, INST_N=8, INST_K=16,
                stages=3,  # Safe default
                swizzle_policy=SwizzlePolicy.Identity, 
                SwizzleN=1
            )
            phase_1_configs.append(cfg)

        # Enqueue all Phase 1
        for cfg in phase_1_configs:
            self.input_queue.put(cfg)
            self.visited_configs.add(cfg.kernel_id())

        # Wait for Phase 1 to complete
        pending = len(phase_1_configs)
        best_tile_config = None
        best_phase1_tflop = 0.0

        start_time = time.time()
        
        while pending > 0:
            if time.time() - start_time > timeout_s: break
            try:
                res = self.output_queue.get(timeout=1.0)
                pending -= 1
                
                # Parse result (simplified logic from your original loop)
                output_path = res.get_output_file_path()
                if output_path and os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as f:
                        perf = KernelPerformance(parse_ncu_log(f.read()))
                    
                    tflops = self.get_tflop_from_runtime(perf.duration_ms)
                    if tflops > best_phase1_tflop:
                        best_phase1_tflop = tflops
                        best_tile_config = res
                        print(f"[PHASE 1] New Best Tile: {res.kernel_id()} ({tflops:.2f} TF)", file=sys.stderr)
            except BaseException:
                continue

        if not best_tile_config:
            print("[ERROR] Phase 1 failed to find any valid config.", file=sys.stderr)
            return

        # --- PHASE 2: Fine Tuning (Stages & Swizzle) ---
        print(f"[LOG] [AUTOTUNER] Phase 2: Refiming Best Tile {best_tile_config.kernel_id()}...", file=sys.stderr)
        
        phase_2_configs = []
        # Use the WINNING Tile Shape
        tb_m, tb_n, tb_k = best_tile_config.TB_M, best_tile_config.TB_N, best_tile_config.TB_K
        w_m, w_n, w_k = best_tile_config.W_M, best_tile_config.W_N, best_tile_config.W_K

        # Exhaustive search of Stages and Swizzle for this specific tile
        for stages in self.STAGES_LIST:
            for swizzle in self.SWIZZLE_FUNCS:
                for swiz_n in self.SWIZZLE_N_VALUES:
                    cfg = KernelConfig(
                        TB_M=tb_m, TB_N=tb_n, TB_K=tb_k,
                        W_M=w_m, W_N=w_n, W_K=w_k,
                        INST_M=16, INST_N=8, INST_K=16,
                        stages=stages,
                        swizzle_policy=swizzle,
                        SwizzleN=swiz_n
                    )
                    if cfg.kernel_id() not in self.visited_configs:
                        phase_2_configs.append(cfg)
        
        # Run Phase 2
        for cfg in phase_2_configs:
            self.input_queue.put(cfg)
        
        # Standard wait loop for Phase 2 results...
        # (You can reuse the generic result loop here)