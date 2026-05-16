import pytest
import numpy as np
from graph import Graph, ErdosRenyiGraph


class ConcreteGraph(Graph):
    def generate(self, *args, **kwargs):
        pass


class TestGraphInit:
    def test_node_neighbors_empty_on_init(self):
        g = ConcreteGraph()
        assert g.node_neighbors == {}


class TestAddNode:
    def test_add_single_node(self):
        g = ConcreteGraph()
        g.add_node(0)
        assert 0 in g.node_neighbors

    def test_add_node_has_empty_neighbors(self):
        g = ConcreteGraph()
        g.add_node(0)
        assert g.node_neighbors[0] == []

    def test_add_multiple_nodes(self):
        g = ConcreteGraph()
        for i in range(5):
            g.add_node(i)
        assert set(g.node_neighbors.keys()) == {0, 1, 2, 3, 4}

    def test_add_duplicate_node_does_not_overwrite(self):
        g = ConcreteGraph()
        g.add_node(0)
        g.node_neighbors[0].append(99)
        g.add_node(0)
        assert 99 in g.node_neighbors[0]


class TestGetNeighbors:
    def test_get_neighbors_empty(self):
        g = ConcreteGraph()
        g.add_node(0)
        assert g.get_neighbors(0) == []

    def test_get_neighbors_after_edge(self):
        g = ConcreteGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1)
        assert 1 in g.get_neighbors(0)
        assert 0 in g.get_neighbors(1)

    def test_get_neighbors_nonexistent_node_raises_key_error(self):
        g = ConcreteGraph()
        with pytest.raises(KeyError):
            g.get_neighbors(42)

    def test_get_neighbors_returns_list(self):
        g = ConcreteGraph()
        g.add_node(0)
        assert isinstance(g.get_neighbors(0), list)


class TestAddEdge:
    def test_add_edge_creates_bidirectional_connection(self):
        g = ConcreteGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1)
        assert 1 in g.node_neighbors[0]
        assert 0 in g.node_neighbors[1]

    def test_add_edge_missing_first_node_raises_value_error(self):
        g = ConcreteGraph()
        g.add_node(1)
        with pytest.raises(ValueError):
            g.add_edge(0, 1)

    def test_add_edge_missing_second_node_raises_value_error(self):
        g = ConcreteGraph()
        g.add_node(0)
        with pytest.raises(ValueError):
            g.add_edge(0, 1)

    def test_add_edge_both_nodes_missing_raises_value_error(self):
        g = ConcreteGraph()
        with pytest.raises(ValueError):
            g.add_edge(0, 1)

    def test_add_multiple_edges(self):
        g = ConcreteGraph()
        for i in range(3):
            g.add_node(i)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        assert 1 in g.node_neighbors[0]
        assert 2 in g.node_neighbors[0]


class TestErdosRenyiGenerate:
    def test_generate_returns_erdos_renyi_instance(self):
        rng = np.random.default_rng(42)
        g = ErdosRenyiGraph.generate(10, 0.9, rng)
        assert isinstance(g, ErdosRenyiGraph)

    def test_generate_correct_number_of_nodes(self):
        rng = np.random.default_rng(42)
        g = ErdosRenyiGraph.generate(10, 0.9, rng)
        assert len(g.node_neighbors) == 10

    def test_generate_nodes_are_zero_indexed(self):
        rng = np.random.default_rng(42)
        g = ErdosRenyiGraph.generate(5, 0.9, rng)
        assert set(g.node_neighbors.keys()) == {0, 1, 2, 3, 4}

    def test_generate_p_below_threshold_raises_value_error(self):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError):
            ErdosRenyiGraph.generate(10, 0.0, rng)

    def test_generate_p_at_threshold_raises_value_error(self):
        import math
        rng = np.random.default_rng(42)
        n = 10
        threshold = math.log(n) / n
        with pytest.raises(ValueError):
            ErdosRenyiGraph.generate(n, threshold - 1e-9, rng)

    def test_generate_p_1_fully_connected(self):
        rng = np.random.default_rng(42)
        n = 5
        g = ErdosRenyiGraph.generate(n, 1.0, rng)
        for node in range(n):
            neighbors = g.get_neighbors(node)
            expected = set(range(n)) - {node}
            assert set(neighbors) == expected

    def test_generate_reproducible_with_same_seed(self):
        g1 = ErdosRenyiGraph.generate(10, 0.9, np.random.default_rng(0))
        g2 = ErdosRenyiGraph.generate(10, 0.9, np.random.default_rng(0))
        assert g1.node_neighbors == g2.node_neighbors

    def test_generate_edges_are_bidirectional(self):
        rng = np.random.default_rng(42)
        g = ErdosRenyiGraph.generate(10, 0.9, rng)
        for node, neighbors in g.node_neighbors.items():
            for neighbor in neighbors:
                assert node in g.node_neighbors[neighbor]