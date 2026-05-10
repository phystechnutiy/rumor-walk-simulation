from abc import ABC, abstractmethod
from itertools import combinations
from agent import Agent
import math as m
import numpy as np


class Graph(ABC):
    """
    Abstract base class for graph structures.
    """

    def __init__(self) -> None:
        """Initialize an empty graph."""
        self.node_neighbors: dict[int, list[int]] = {}

    def get_neighbors(self, node: int) -> list[int]:
        """
        Return the neighbors of a given node.

        Raises:
            KeyError: If the node does not exist.
        """
        if node not in self.node_neighbors.keys():
            raise KeyError
        return self.node_neighbors[node]

    def add_node(self, node: int) -> None:
        """Add a node to the graph."""
        if node not in self.node_neighbors.keys():
            self.node_neighbors[node] = []

    def add_edge(self, node1: int, node2: int) -> None:
        """
        Add an undirected edge between two nodes.

        Raises:
            KeyError: If either node does not exist.
        """
        if node1 not in self.node_neighbors.keys() or node2 not in self.node_neighbors.keys():
            raise ValueError
        self.node_neighbors[node1].append(node2)
        self.node_neighbors[node2].append(node1)

    @abstractmethod
    def generate(self, *args, **kwargs) -> None:
        """Generate a graph instance."""
        pass


class ErdosRenyiGraph(Graph):
    """
    Erdős–Rényi random graph model.
    """

    @classmethod
    def generate(cls, n: int, p: float, rng: np.random.Generator) -> "ErdosRenyiGraph":
        """
        Generate an Erdős–Rényi graph.

        Args:
            n (int): Number of nodes.
            p (float): Probability of edge creation.
            rng (np.random.Generator): Random number generator.

        Raises:
            KeyError: If probability is below the connectivity threshold.

        Returns:
            ErdosRenyiGraph: Generated graph instance.
        """
        if p < m.log(n, m.e) / n:
            raise ValueError

        instance = cls()

        for node in range(n):
            instance.add_node(node)

        for node1, node2 in combinations(range(n), 2):
            if rng.random() < p:
                instance.add_edge(node1, node2)

        return instance
