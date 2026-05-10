from graph import ErdosRenyiGraph
from agent import Agent, State
from interaction_model import InteractionModel
from simulator import Simulator
import random
import numpy as np
from phase_analyzer import PhaseAnalyzer

rng = np.random.default_rng(seed=0)

def build_simulator(n: int) -> Simulator:
    # graph
    graph = ErdosRenyiGraph.generate(n, p=0.5, rng=rng)

    # agents
    agents = [Agent(State.ignorant, position=random.randint(0, n-1)) for _ in range(n)]

    # one spreader
    agents[0].state = State.spreader

    # node_occupants
    node_occupants = {i: [] for i in range(n)}
    for a in agents:
        node_occupants[a.position].append(a)

    # interaction model
    interaction_model = InteractionModel(_lambda=0.3, _alpha=0.3, rng=rng)

    # simulator
    return Simulator(
        agents=agents,
        graph=graph,
        node_occupants=node_occupants,
        interaction_model=interaction_model,
        rng=rng
    )

def _monte_carlo():
    sim = build_simulator(3)

    results = sim.run_monte_carlo(n_runs=5)

    print("Monte Carlo results:\n")

    for i, run in enumerate(results):
        print(f"\nRun {i}:")
        for row in run:
            print(row)


def init_phase_analyzer() -> PhaseAnalyzer:
    analyzer = PhaseAnalyzer(
        lambda_start=0.1,
        lambda_step=0.1,
        sizes=[10, 50],
        simulator_factory=build_simulator,
        crit_finder=None
    )
    return analyzer

def _critical_lambdas():
    analyzer = init_phase_analyzer()
    analyzer.crit_finder = analyzer.find_param_crit
    results = analyzer.run(n_runs=10)
    print(results)

def _inflection_points():
    analyzer = init_phase_analyzer()
    analyzer.crit_finder = analyzer.find_inflection_point
    results = analyzer.run(n_runs=10)
    print(results)

def main():
    pass

if __name__ == "__main__":
    main()
