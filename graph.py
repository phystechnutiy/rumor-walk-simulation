from abc import ABC, abstractmethod
from itertools import combinations
from rng import rng

class Graph(ABC):
    def __init__(self):
        self.node_neighbors = {}

    def get_neighbors(self, node_id):
        if node_id not in self.node_neighbors.keys():
            raise KeyError
        return self.node_neighbors[node_id]

    def add_node(self, node):
        if node not in self.node_neighbors.keys():
            self.node_neighbors[node] = []
        return

    def add_edge(self, node1, node2):
        if node1 not in self.node_neighbors.keys() or node2 not in self.node_neighbors.keys():
            raise KeyError
        self.node_neighbors[node1].append(node2)
        self.node_neighbors[node2].append(node1)

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass



class ErdosRenyiGraph(Graph):
    @classmethod
    def generate(cls, n, p):
        instance = cls()

        for node in range(n):
            instance.add_node(node)

        for node1, node2 in combinations(range(n), 2):
            if rng.random() < p:
                instance.add_edge(node1, node2)

        return instance
