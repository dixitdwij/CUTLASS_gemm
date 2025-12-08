from enum import Enum
from dataclasses import dataclass


class SwizzlePolicy(Enum):
    Identity = "SwizzleIdentity"
    SplitK = "SwizzleSplitK"
    Horizontal = "SwizzleHorizontal"


@dataclass
class KernelConfig:
    def __init__(
        self,
        TB_M: int, TB_N: int, TB_K: int,
        W_M: int, W_N: int, W_K: int,
        INST_M: int, INST_N: int, INST_K: int,
        swizzle_policy: SwizzlePolicy,
        SwizzleN: int
    ):
        self.TB_M = TB_M
        self.TB_N = TB_N
        self.TB_K = TB_K

        self.W_M = W_M
        self.W_N = W_N
        self.W_K = W_K

        self.INST_M = INST_M
        self.INST_N = INST_N
        self.INST_K = INST_K

        self.swizzle_policy = swizzle_policy
        self.SwizzleN = SwizzleN

    def compilation_flags(self):
        tokens = [
            f"-DTB_M={self.TB_M}",
            f"-DTB_N={self.TB_N}",
            f"-DTB_K={self.TB_K}",
            f"-DW_M={self.W_M}",
            f"-DW_N={self.W_N}",
            f"-DW_K={self.W_K}",
            f"-DINST_M={self.INST_M}",
            f"-DINST_N={self.INST_N}",
            f"-DINST_K={self.INST_K}",
            f"-D{self.swizzle_policy.value}",
            f"-DSwizzleN={self.SwizzleN}"
        ]
        return " ".join(tokens)
    
    def kernel_id(self):
        return (
            f"TB_{self.TB_M}x{self.TB_N}x{self.TB_K}"
            f"__W_{self.W_M}x{self.W_N}x{self.W_K}"
            f"__INST_{self.INST_M}x{self.INST_N}x{self.INST_K}"
            f"__SWZ_{self.swizzle_policy.name}_N{self.SwizzleN}"
        )
    
    @staticmethod
    def from_id(id_str: str) -> "KernelConfig":
        segments = id_str.split("__")
        seg_map = {}
        for seg in segments:
            key, value = seg.split("_", 1)
            seg_map[key] = value

        tb_vals = list(map(int, seg_map["TB"].split("x")))
        w_vals = list(map(int, seg_map["W"].split("x")))
        inst_vals = list(map(int, seg_map["INST"].split("x")))

        # SWZ: "SplitK_N2" or "Identity_N1" etc.
        swz_part = seg_map["SWZ"]  # e.g. "SplitK_N2"
        swz_name_str, n_str = swz_part.split("_N")
        swizzle_policy = SwizzlePolicy[swz_name_str]
        swizzle_n = int(n_str)

        return KernelConfig(
            TB_M=tb_vals[0], TB_N=tb_vals[1], TB_K=tb_vals[2],
            W_M=w_vals[0], W_N=w_vals[1], W_K=w_vals[2],
            INST_M=inst_vals[0], INST_N=inst_vals[1], INST_K=inst_vals[2],
            swizzle_policy=swizzle_policy,
            SwizzleN=swizzle_n,
        )
    
    def register_compilation(self, bin_path: str, compilation_successful: bool) -> None:
        self.binary_path = bin_path
        self.compiled = compilation_successful

    def is_compiled(self) -> bool:
        return getattr(self, 'compiled', False)
    
    def get_binary_path(self) -> str:
        return self.binary_path  # Error catching intended
    
    def register_output_file(self, output_file: str) -> None:
        self.output_file = output_file

    def get_output_file_path(self) -> str:
        return self.output_file # Error catching intended
