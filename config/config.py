from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class SimulationConfig:
    mode: str = "monte_carlo"
    n: int = 10
    p: float = 0.5
    lambda_: float = 0.3
    alpha: float = 0.3
    seed: int = 0
    n_runs: int = 5
    lambda_start: float = 0.1
    lambda_step: float = 0.1
    sizes: list[int] = field(default_factory=lambda: [10, 50])

cs = ConfigStore.instance()
cs.store(name="config", node=SimulationConfig)

