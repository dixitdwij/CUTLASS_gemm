from abc import ABC, abstractmethod
from typing import List, Optional
from ncu_parser import KernelPerformance

class ActionType:
    TB_TILE = "TB_TILE"
    STAGES = "STAGES"
    WARP_DIV = "WARP_DIV"
    SWIZZLE_N = "SWIZZLE_N"
    SWIZZLE_POLICY = "SWIZZLE_POLICY" # Toggle Identity/SplitK
    RANDOM_WARP = "RANDOM_WARP"     # Perturb warp shape

class Action:
    def __init__(self, param: str, direction: int = 0):
        self.param = param
        self.direction = direction # +1 (increase), -1 (decrease), 0 (toggle/random)

    def __repr__(self):
        dir_str = "+1" if self.direction > 0 else "-1" if self.direction < 0 else "0"
        return f"Action({self.param}, {dir_str})"

class Policy(ABC):
    @abstractmethod
    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        """
        Returns an Action if the policy condition is met, otherwise None.
        """
        pass

class CompositePolicy(Policy):
    """
    Composes multiple policies. Evaluates them in order and returns the action 
    of the first triggered policy.
    """
    def __init__(self, policies: List[Policy]):
        self.policies = policies

    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        for policy in self.policies:
            action = policy.evaluate(perf)
            if action:
                return action
        return None

# --- Specific Policies ---

class HighRegisterPressure(Policy):
    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        # Condition: reg_per_thread >= 240
        if perf.reg_per_thread >= 240:
            return Action(ActionType.TB_TILE, -1) # DECREASE_TILE_SIZE
        return None

class L1CacheThrashing(Policy):
    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        # Condition: l1_tex_hit_rate < 40.0
        if perf.l1_tex_hit_rate_pct < 40.0:
            return Action(ActionType.TB_TILE, -1) # DECREASE_TILE_SIZE
        return None

class ResourceExhaustion(Policy):
    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        # Condition: reg > 200 OR sm < 30
        if perf.reg_per_thread > 200 or perf.sm__pct < 30.0:
            return Action(ActionType.STAGES, -1) # DECREASE_STAGES
        return None

class ComputeBound(Policy):
    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        # Condition: sm_pct > 80 AND mem < 60
        if perf.sm__pct > 80.0 and perf.mem_throughput_pct < 60.0:
            return Action(ActionType.TB_TILE, 1) # INCREASE_TILE_SIZE
        return None

class MemoryBound(Policy):
    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        # Condition: mem > 80 AND sm < 60
        if perf.mem_throughput_pct > 80.0 and perf.sm__pct < 60.0:
            return Action(ActionType.TB_TILE, 1) # INCREASE_TILE_SIZE (Increase reuse)
        return None

class LatencyBound(Policy):
    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        # Condition: sm < 40 AND mem < 40
        if perf.sm__pct < 40.0 and perf.mem_throughput_pct < 40.0:
            return Action(ActionType.TB_TILE, -1) # DECREASE_TILE_SIZE
        return None

class HighLatencySensitivity(Policy):
    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        # Condition: mem > 70 AND dram > 70
        if perf.mem_throughput_pct > 70.0 and perf.dram_throughput_pct > 70.0:
            return Action(ActionType.STAGES, 1) # INCREASE_STAGES
        return None

class PartitionCamping(Policy):
    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        # Condition: dram < 50 AND mem > 75
        if perf.dram_throughput_pct < 50.0 and perf.mem_throughput_pct > 75.0:
            return Action(ActionType.SWIZZLE_POLICY, 0) # CHANGE_SWIZZLE_POLICY
        return None

class L2CachePollution(Policy):
    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        # Condition: l2_hit_rate < 40.0
        if perf.l2_hit_rate_pct < 40.0:
            return Action(ActionType.STAGES, -1) # DECREASE_STAGES
        return None

class PoorSpatialLocality(Policy):
    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        # Condition: l2_hit_rate < 50.0
        if perf.l2_hit_rate_pct < 50.0:
            return Action(ActionType.SWIZZLE_N, 1) # INCREASE_SWIZZLEN
        return None

class DependencyStalls(Policy):
    def evaluate(self, perf: KernelPerformance) -> Optional[Action]:
        # Condition: ipc < 1.0 AND sm < 50.0
        if perf.ipc < 1.0 and perf.sm__pct < 50.0:
            return Action(ActionType.RANDOM_WARP, 0) # RANDOM_PERTURB
        return None

def get_default_policy() -> Policy:
    # Composes policies in the order of criticality
    return CompositePolicy([
        HighRegisterPressure(),
        L1CacheThrashing(),
        ResourceExhaustion(),
        ComputeBound(),
        MemoryBound(),
        LatencyBound(),
        HighLatencySensitivity(),
        PartitionCamping(),
        L2CachePollution(),
        PoorSpatialLocality(),
        DependencyStalls()
    ])
