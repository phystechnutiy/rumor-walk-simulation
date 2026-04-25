from agent import Agent, State
from graph import Graph
from interaction_model import InteractionModel
from itertools import combinations
from constants import *
from concurrent.futures import ProcessPoolExecutor
import copy
import numpy as np


class Simulator:
    """
    Runs the rumor spreading simulation.
    """

    def __init__(
        self,
        agents: list[Agent],
        graph: Graph,
        node_occupants: dict[int, list[Agent]],
        interaction_model: InteractionModel,
        rng: np.random.Generator
    ) -> None:
        """
        Initialize the simulator.

        Args:
            agents (list[Agent]): List of agents in the simulation.
            graph (Graph): Graph defining movement between nodes.
            node_occupants (dict[int, list[Agent]]): Mapping of nodes to agents.
            interaction_model (InteractionModel): Interaction rules.
            rng (np.random.Generator): Random number generator.
        """
        self.agents = agents
        self.graph = graph
        self.node_occupants = node_occupants
        self.stats = []
        self.snapshots = []
        self.interaction_model = interaction_model
        self.rng = rng

        self._initial_agents = copy.deepcopy(agents)
        self._initial_node_occupants = copy.deepcopy(node_occupants)

    def collect(self, tick: int) -> None:
        """
        Collect statistics and snapshot at a given tick.
        """
        ignorant = 0
        spreader = 0
        stifler = 0

        for agent in self.agents:
            if agent.state == State.ignorant:
                ignorant += 1
            elif agent.state == State.spreader:
                spreader += 1
            elif agent.state == State.stifler:
                stifler += 1

        self.stats.append({
            "tick": tick,
            "ignorant": ignorant,
            "spreader": spreader,
            "stifler": stifler
        })

        self.snapshots.append({
            "tick": tick,
            "node_occupants": copy.deepcopy(self.node_occupants)
        })

    def step(self) -> None:
        """
        Perform a single simulation step:
        interactions followed by agent movement.
        """
        for node in self.node_occupants.keys():
            for a, b in combinations(self.node_occupants[node], 2):
                self.interaction_model.interact(a, b)

        for agent in self.agents:
            new_position = int(self.rng.choice(self.graph.get_neighbors(agent.position)))
            self.node_occupants[agent.position].remove(agent)
            agent.position = new_position
            if new_position not in self.node_occupants.keys():
                self.node_occupants[new_position] = []
            self.node_occupants[agent.position].append(agent)

    def run(self) -> None:
        """
        Run the simulation until no spreaders remain.
        """
        tick = 0

        while any(agent.state == State.spreader for agent in self.agents):
            self.step()
            self.collect(tick)
            tick += 1

    def reset(self) -> None:
        """
        Reset the simulation to its initial state.
        """
        self.stats = []
        self.snapshots = []
        self.agents = copy.deepcopy(self._initial_agents)
        self.node_occupants = {}

        for agent in self.agents:
            if agent.position not in self.node_occupants.keys():
                self.node_occupants[agent.position] = [agent]
            else:
                self.node_occupants[agent.position].append(agent)

    def _single_run(self) -> list:
        """
        Execute a single simulation run.

        Returns:
            list: Collected statistics for the run.
        """
        self.rng = np.random.default_rng()
        self.interaction_model.rng = self.rng
        self.reset()
        self.run()
        return copy.deepcopy(self.stats)

    def run_monte_carlo(self, n_runs: int) -> list:
        """
        Run multiple independent simulations in parallel.

        Args:
            n_runs (int): Number of runs.

        Returns:
            list: List of statistics from each run.
        """
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._single_run) for _ in range(n_runs)]
            return [f.result() for f in futures]