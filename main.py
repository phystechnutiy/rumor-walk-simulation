from config.config import SimulationConfig
from graph import ErdosRenyiGraph
from agent import Agent, State
from interaction_model import InteractionModel
from simulator import Simulator
from dashboard import SimulationDashboard
import numpy as np
from phase_analyzer import PhaseAnalyzer
from typing import Callable
import hydra


def build_simulator(n: int, rng: np.random.Generator, p: float, spread_prob: float, stifle_prob: float, cooperate_prob: float) -> tuple[Simulator, ErdosRenyiGraph]:
    """
    Build and return a configured Simulator and graph instance.
    Args:
        n (int): Number of agents and nodes.
        rng (np.random.Generator): Random number generator.
        p (float): Edge probability for Erdos-Renyi graph.
        spread_prob (float): Probability of spreading the rumor.
        stifle_prob (float): Probability of becoming a stifler.
        cooperate_prob (float): Probability of becoming a cooperator.
    Returns:
        tuple[Simulator, ErdosRenyiGraph]: Configured simulator and graph.
    """
    graph = ErdosRenyiGraph.generate(n, p, rng)
    agents = [Agent(State.ignorant, int(rng.integers(0, n - 1))) for _ in range(n)]
    agents[0].state = State.spreader
    node_occupants = {i: [] for i in range(n)}
    for a in agents:
        node_occupants[a.position].append(a)
    interaction_model = InteractionModel(spread_prob=spread_prob, stifle_prob=stifle_prob, cooperate_prob=cooperate_prob, rng=rng)
    sim = Simulator(agents=agents, graph=graph, node_occupants=node_occupants, interaction_model=interaction_model, rng=rng)
    return sim, graph


def monte_carlo(n: int, n_runs: int, rng: np.random.Generator, p: float, spread_prob: float, stifle_prob: float, cooperate_prob: float, dashboard: bool = False) -> None:
    """
    Run Monte Carlo simulation and print results.
    Args:
        n (int): Number of agents and nodes.
        n_runs (int): Number of simulation runs.
        rng (np.random.Generator): Random number generator.
        p (float): Edge probability.
        spread_prob (float): Probability of spreading the rumor.
        stifle_prob (float): Probability of becoming a stifler.
        cooperate_prob (float): Probability of becoming a cooperator.
        dashboard (bool): Whether to launch the visual dashboard.
    """
    sim, graph = build_simulator(n, rng, p, spread_prob, stifle_prob, cooperate_prob)
    if dashboard:
        sim.run()
        SimulationDashboard(sim, graph).run()
    else:
        results = sim.run_monte_carlo(n_runs)
        print("Monte Carlo results:\n")
        for i, run in enumerate(results):
            print(f"\nRun {i}:")
            for row in run:
                print(row)


def init_phase_analyzer(rng: np.random.Generator, p: float, spread_prob: float, stifle_prob: float, cooperate_prob: float, param_start: float, param_step: float, sizes: list[int], crit_finder: Callable[[dict[float, float]], float]) -> PhaseAnalyzer:
    """
    Initialize and return a PhaseAnalyzer instance.
    Args:
        rng (np.random.Generator): Random number generator.
        p (float): Edge probability.
        spread_prob (float): Probability of spreading the rumor.
        stifle_prob (float): Probability of becoming a stifler.
        cooperate_prob (float): Probability of becoming a cooperator.
        param_start (float): Starting value of lambda sweep.
        param_step (float): Step size of lambda sweep.
        sizes (list[int]): List of system sizes to analyze.
        crit_finder (Callable): Function to find critical lambda.
    Returns:
        PhaseAnalyzer: Configured phase analyzer.
    """
    return PhaseAnalyzer(
        lambda_start=param_start,
        lambda_step=param_step,
        sizes=sizes,
        simulator_factory=lambda n: build_simulator(n, rng, p, spread_prob, stifle_prob, cooperate_prob)[0],
        crit_finder=crit_finder
    )


def critical_lambdas(n_runs: int, rng: np.random.Generator, p: float, spread_prob: float, stifle_prob: float, cooperate_prob: float, param_start: float, param_step: float, sizes: list[int], crit_finder: Callable[[dict[float, float]], float]) -> None:
    """
    Run phase analysis and print critical lambda values.
    Args:
        n_runs (int): Number of Monte Carlo runs per lambda value.
        rng (np.random.Generator): Random number generator.
        p (float): Edge probability.
        spread_prob (float): Probability of spreading the rumor.
        stifle_prob (float): Probability of becoming a stifler.
        cooperate_prob (float): Probability of becoming a cooperator.
        param_start (float): Starting value of lambda sweep.
        param_step (float): Step size of lambda sweep.
        sizes (list[int]): List of system sizes to analyze.
        crit_finder (Callable): Function to find critical lambda.
    """
    analyzer = init_phase_analyzer(rng, p, spread_prob, stifle_prob, cooperate_prob, param_start, param_step, sizes, crit_finder)
    analyzer.crit_finder = analyzer.find_param_crit
    results = analyzer.run(n_runs)
    print(results)


def inflection_points(n_runs: int, rng: np.random.Generator, p: float, spread_prob: float, stifle_prob: float, cooperate_prob: float, param_start: float, param_step: float, sizes: list[int], crit_finder: Callable[[dict[float, float]], float]) -> None:
    """
    Run phase analysis and print inflection point lambda values.
    Args:
        n_runs (int): Number of Monte Carlo runs per lambda value.
        rng (np.random.Generator): Random number generator.
        p (float): Edge probability.
        spread_prob (float): Probability of spreading the rumor.
        stifle_prob (float): Probability of becoming a stifler.
        cooperate_prob (float): Probability of becoming a cooperator.
        param_start (float): Starting value of lambda sweep.
        param_step (float): Step size of lambda sweep.
        sizes (list[int]): List of system sizes to analyze.
        crit_finder (Callable): Function to find critical lambda.
    """
    analyzer = init_phase_analyzer(rng, p, spread_prob, stifle_prob, cooperate_prob, param_start, param_step, sizes, crit_finder)
    analyzer.crit_finder = analyzer.find_inflection_point
    results = analyzer.run(n_runs)
    print(results)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: SimulationConfig) -> None:
    """Entry point for the simulation."""
    rng = np.random.default_rng(seed=cfg.seed)
    if cfg.mode == "monte_carlo":
        monte_carlo(n=cfg.n, n_runs=cfg.n_runs, rng=rng, p=cfg.p, spread_prob=cfg.spread_prob, stifle_prob=cfg.stifle_prob, cooperate_prob=cfg.cooperate_prob, dashboard=cfg.dashboard)
    elif cfg.mode == "critical":
        critical_lambdas(n_runs=cfg.n_runs, rng=rng, p=cfg.p, spread_prob=cfg.spread_prob, stifle_prob=cfg.stifle_prob, cooperate_prob=cfg.cooperate_prob, param_start=cfg.param_start, param_step=cfg.param_step, sizes=cfg.sizes)
    elif cfg.mode == "inflection":
        inflection_points(n_runs=cfg.n_runs, rng=rng, p=cfg.p, spread_prob=cfg.spread_prob, stifle_prob=cfg.stifle_prob, cooperate_prob=cfg.cooperate_prob, param_start=cfg.param_start, param_step=cfg.param_step, sizes=cfg.sizes)


if __name__ == "__main__":
    main()