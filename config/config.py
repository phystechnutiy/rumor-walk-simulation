from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class SimulationConfig:
    mode: str = "monte_carlo"
    n: int = 10
    p: float = 0.5
    spread_prob: float = 0.3
    stifle_prob: float = 0.3
    cooperate_prob: float = 0.1
    seed: int = 0
    n_runs: int = 5
    param_start: float = 0.1
    param_step: float = 0.1
    sizes: list[int] = field(default_factory=lambda: [10, 50])

cs = ConfigStore.instance()
cs.store(name="config", node=SimulationConfig)