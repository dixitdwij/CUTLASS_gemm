import os
import subprocess
import itertools
import re
import sys
from typing import List, Dict, Tuple, Optional

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

    def __init__(self, source_file="dyntuned_gemm.cu", binary_path="bin/dyntuned_gemm"):
        self.source_file = source_file
        self.binary_path = binary_path
        self.results = []

    