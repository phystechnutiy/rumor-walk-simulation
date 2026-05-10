from config.config import SimulationConfig
from graph import ErdosRenyiGraph
from agent import Agent, State
from interaction_model import InteractionModel
from simulator import Simulator
import numpy as np
from phase_analyzer import PhaseAnalyzer
from typing import Callable
import hydra




def build_simulator(n: int, rng: np.random.Generator, p: float, lamb: float, alpha: float) -> Simulator:
    # graph
    graph = ErdosRenyiGraph.generate(n, p, rng)

    # agents
    agents = [Agent(State.ignorant, int(rng.integers(0, n-1))) for _ in range(n)]

    # one spreader
    agents[0].state = State.spreader

    # node_occupants
    node_occupants = {i: [] for i in range(n)}
    for a in agents:
        node_occupants[a.position].append(a)

    # interaction model
    interaction_model = InteractionModel(_lambda=lamb, _alpha=alpha, rng=rng)

    return Simulator(
        agents=agents,
        graph=graph,
        node_occupants=node_occupants,
        interaction_model=interaction_model,
        rng=rng
    )

def monte_carlo(n: int, n_runs: int, rng: np.random.Generator, p: float, lamb: float, alpha: float) -> float:
    sim = build_simulator(n, rng, p, lamb, alpha)

    results = sim.run_monte_carlo(n_runs)

    print("Monte Carlo results:\n")

    for i, run in enumerate(results):
        print(f"\nRun {i}:")
        for row in run:
            print(row)


def init_phase_analyzer(rng: np.random.Generator, p: float, lamb: float, alpha: float, lambda_start: float, lambda_step: float, sizes: list[int], crit_finder: Callable[[dict[float, float]], float]) -> PhaseAnalyzer:
    analyzer = PhaseAnalyzer(
        lambda_start = lambda_start,
        lambda_step = lambda_step,
        sizes = sizes,
        simulator_factory = lambda n: build_simulator(n, rng, p, lamb, alpha),
        crit_finder = crit_finder
    )
    return analyzer

def critical_lambdas(n_runs: int, rng: np.random.Generator, p: float, lamb: float, alpha: float, lambda_start: float, lambda_step: float, sizes: list[int], crit_finder: Callable[[dict[float, float]], float]):
    analyzer = init_phase_analyzer(rng, p, lamb, alpha, lambda_start, lambda_step, sizes, crit_finder)
    analyzer.crit_finder = analyzer.find_param_crit
    results = analyzer.run(n_runs)
    print(results)

def inflection_points(n_runs: int, rng: np.random.Generator, p: float, lamb: float, alpha: float, lambda_start: float, lambda_step: float, sizes: list[int], crit_finder: Callable[[dict[float, float]], float]):
    analyzer = init_phase_analyzer(rng, p, lamb, alpha, lambda_start, lambda_step, sizes, crit_finder)
    analyzer.crit_finder = analyzer.find_inflection_point
    results = analyzer.run(n_runs)
    print(results)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: SimulationConfig) -> None:
    rng = np.random.default_rng(seed=cfg.seed)

    if cfg.mode == "monte_carlo":
        monte_carlo(n=cfg.n, n_runs=cfg.n_runs, rng=rng, p=cfg.p, lamb=cfg.lambda_, alpha=cfg.alpha)
    elif cfg.mode == "critical":
        critical_lambdas(n_runs=cfg.n_runs, rng=rng, p=cfg.p, lamb=cfg.lambda_, alpha=cfg.alpha, lambda_start=cfg.lambda_start, lambda_step=cfg.lambda_step, sizes=cfg.sizes)
    elif cfg.mode == "inflection":
        inflection_points(n_runs=cfg.n_runs, rng=rng, p=cfg.p, lamb=cfg.lambda_, alpha=cfg.alpha, lambda_start=cfg.lambda_start, lambda_step=cfg.lambda_step, sizes=cfg.sizes)

if __name__ == "__main__":
    main()


