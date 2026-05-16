import pytest
import copy
import numpy as np
from unittest.mock import MagicMock, patch
from agent import Agent, State
from graph import ErdosRenyiGraph
from interaction_model import InteractionModel
from simulator import Simulator


def make_rng(seed=42):
    return np.random.default_rng(seed)


def make_interaction_model(rng, spread=0.5, stifle=0.3, cooperate=0.1):
    return InteractionModel(spread, stifle, cooperate, rng)


def make_graph(n=5, p=1.0, rng=None):
    """Полносвязный граф — p=1.0 гарантирует рёбра между всеми узлами."""
    if rng is None:
        rng = make_rng()
    return ErdosRenyiGraph.generate(n, p, rng)


def make_simulator(n_agents=3, spreader_index=0, seed=42):
    """
    Создаёт минимальный рабочий симулятор:
    - n_agents агентов, один из них spreader, остальные ignorant
    - полносвязный граф на n_agents узлах
    - каждый агент на своём узле
    """
    rng = make_rng(seed)
    graph = make_graph(n=n_agents, p=1.0, rng=make_rng(seed + 1))
    model = make_interaction_model(rng)

    agents = []
    for i in range(n_agents):
        state = State.spreader if i == spreader_index else State.ignorant
        agents.append(Agent(state, i))

    node_occupants = {i: [agents[i]] for i in range(n_agents)}

    return Simulator(agents, graph, node_occupants, model, rng)


def make_simulator_no_spreaders(n_agents=3, seed=42):
    """Все агенты — ignorant. run() должен завершиться мгновенно."""
    rng = make_rng(seed)
    graph = make_graph(n=n_agents, p=1.0, rng=make_rng(seed + 1))
    model = make_interaction_model(rng)
    agents = [Agent(State.ignorant, i) for i in range(n_agents)]
    node_occupants = {i: [agents[i]] for i in range(n_agents)}
    return Simulator(agents, graph, node_occupants, model, rng)


class TestInit:
    def test_agents_stored(self):
        sim = make_simulator()
        assert len(sim.agents) == 3

    def test_stats_empty_on_init(self):
        sim = make_simulator()
        assert sim.stats == []

    def test_snapshots_empty_on_init(self):
        sim = make_simulator()
        assert sim.snapshots == []

    def test_initial_agents_are_deep_copy(self):
        sim = make_simulator()
        sim.agents[0].state = State.stifler
        assert sim._initial_agents[0].state != State.stifler

    def test_initial_node_occupants_are_deep_copy(self):
        sim = make_simulator()
        original_keys = set(sim._initial_node_occupants.keys())
        sim.node_occupants[99] = []
        assert 99 not in sim._initial_node_occupants


class TestCollect:
    def test_collect_appends_one_stat(self):
        sim = make_simulator()
        sim.collect(0)
        assert len(sim.stats) == 1

    def test_collect_tick_stored(self):
        sim = make_simulator()
        sim.collect(7)
        assert sim.stats[0]["tick"] == 7

    def test_collect_counts_ignorant_correctly(self):
        sim = make_simulator(n_agents=3, spreader_index=0)
        sim.collect(0)
        assert sim.stats[0]["ignorant"] == 2

    def test_collect_counts_spreader_correctly(self):
        sim = make_simulator(n_agents=3, spreader_index=0)
        sim.collect(0)
        assert sim.stats[0]["spreader"] == 1

    def test_collect_counts_stifler_correctly(self):
        sim = make_simulator(n_agents=3, spreader_index=0)
        sim.agents[1].state = State.stifler
        sim.collect(0)
        assert sim.stats[0]["stifler"] == 1

    def test_collect_counts_stifler_correctly(self):
        sim = make_simulator(n_agents=3, spreader_index=0)
        sim.agents[1].state = State.cooperator
        sim.collect(0)
        assert sim.stats[0]["cooperator"] == 1

    def test_collect_appends_snapshot(self):
        sim = make_simulator()
        sim.collect(0)
        assert len(sim.snapshots) == 1

    def test_collect_snapshot_is_deep_copy(self):
        sim = make_simulator()
        sim.collect(0)
        snap_before = sim.snapshots[0]["node_occupants"]
        sim.node_occupants[99] = []
        assert 99 not in snap_before

    def test_collect_cooperator_not_counted_in_stats(self):
        sim = make_simulator(n_agents=3, spreader_index=0)
        sim.agents[1].state = State.cooperator
        sim.collect(0)
        stat = sim.stats[0]
        assert stat["ignorant"] + stat["spreader"] + stat["stifler"] == 2

    def test_collect_multiple_ticks(self):
        sim = make_simulator()
        sim.collect(0)
        sim.collect(1)
        assert len(sim.stats) == 2
        assert sim.stats[1]["tick"] == 1


class TestStep:
    def test_step_agents_move(self):
        """После шага хотя бы один агент должен сменить позицию (полносвязный граф)."""
        sim = make_simulator(n_agents=4, seed=0)
        positions_before = [a.position for a in sim.agents]
        sim.step()
        positions_after = [a.position for a in sim.agents]
        assert positions_before != positions_after

    def test_step_node_occupants_consistent_with_agents(self):
        """node_occupants должен соответствовать позициям агентов после шага."""
        sim = make_simulator(n_agents=3, seed=0)
        sim.step()
        for agent in sim.agents:
            assert agent in sim.node_occupants[agent.position]

    def test_step_agent_stays_if_no_neighbors(self):
        """Агент без соседей должен оставаться на месте."""
        rng = make_rng()
        model = make_interaction_model(rng)

        from graph import ErdosRenyiGraph
        graph = ErdosRenyiGraph()
        graph.add_node(0)

        agent = Agent(State.spreader, 0)
        node_occupants = {0: [agent]}

        sim = Simulator([agent], graph, node_occupants, model, rng)
        sim.step()
        assert agent.position == 0

class TestRun:
    def test_run_terminates_when_no_spreaders(self):
        sim = make_simulator_no_spreaders()
        sim.run()
        assert sim.stats == []

    def test_run_terminates_with_one_spreader(self):
        sim = make_simulator(n_agents=5, spreader_index=0, seed=1)
        sim.run()
        assert not any(a.state == State.spreader for a in sim.agents)

    def test_run_collects_stats_each_tick(self):
        sim = make_simulator(n_agents=3, spreader_index=0, seed=2)
        sim.run()
        assert len(sim.stats) > 0

    def test_run_stats_ticks_are_sequential(self):
        sim = make_simulator(n_agents=3, spreader_index=0, seed=2)
        sim.run()
        ticks = [s["tick"] for s in sim.stats]
        assert ticks == list(range(len(ticks)))

class TestReset:
    def test_reset_clears_stats(self):
        sim = make_simulator()
        sim.collect(0)
        sim.reset()
        assert sim.stats == []

    def test_reset_clears_snapshots(self):
        sim = make_simulator()
        sim.collect(0)
        sim.reset()
        assert sim.snapshots == []

    def test_reset_restores_agent_states(self):
        sim = make_simulator(n_agents=3, spreader_index=0)
        initial_states = [a.state for a in sim._initial_agents]
        sim.run()
        sim.reset()
        restored_states = [a.state for a in sim.agents]
        assert restored_states == initial_states

    def test_reset_restores_agent_positions(self):
        sim = make_simulator(n_agents=3, spreader_index=0)
        initial_positions = [a.position for a in sim._initial_agents]
        sim.run()
        sim.reset()
        restored_positions = [a.position for a in sim.agents]
        assert restored_positions == initial_positions

    def test_reset_rebuilds_node_occupants(self):
        sim = make_simulator(n_agents=3, spreader_index=0)
        sim.run()
        sim.reset()
        for agent in sim.agents:
            assert agent in sim.node_occupants[agent.position]

    def test_reset_allows_rerun(self):
        sim = make_simulator(n_agents=3, spreader_index=0, seed=5)
        sim.run()
        stats_first = copy.deepcopy(sim.stats)
        sim.reset()
        sim.run()
        assert len(sim.stats) > 0

class TestRunMonteCarlo:
    def test_returns_correct_number_of_runs(self):
        sim = make_simulator(n_agents=3, spreader_index=0, seed=0)
        results = sim.run_monte_carlo(3)
        assert len(results) == 3

    def test_each_run_contains_stats(self):
        sim = make_simulator(n_agents=3, spreader_index=0, seed=0)
        results = sim.run_monte_carlo(2)
        for run in results:
            assert isinstance(run, list)

    def test_each_stat_has_required_keys(self):
        sim = make_simulator(n_agents=3, spreader_index=0, seed=0)
        results = sim.run_monte_carlo(2)
        for run in results:
            for stat in run:
                assert "tick" in stat
                assert "ignorant" in stat
                assert "spreader" in stat
                assert "stifler" in stat
                assert "cooperator" in stat


class TestStepWithMultipleAgentsSameNode:
    def test_step_multiple_interactions_on_same_node(self):
        rng = make_rng(42)
        from graph import ErdosRenyiGraph
        graph = ErdosRenyiGraph()
        graph.add_node(0)

        model = make_interaction_model(rng, spread=1.0, stifle=0.0, cooperate=0.0)

        agents = [
            Agent(State.spreader, 0),
            Agent(State.ignorant, 1),
            Agent(State.ignorant, 2)
        ]

        for agent in agents:
            agent.position = 0

        node_occupants = {0: agents}

        sim = Simulator(agents, graph, node_occupants, model, rng)

        sim.step()

        spreader_count = sum(1 for a in agents if a.state == State.spreader)
        assert spreader_count == 3


class TestResetMultipleTimes:
    def test_reset_multiple_times_works_correctly(self):
        sim = make_simulator(n_agents=3, spreader_index=0, seed=42)

        sim.run()
        first_run_stats_length = len(sim.stats)
        sim.reset()
        sim.run()
        second_run_stats_length = len(sim.stats)

        assert first_run_stats_length == second_run_stats_length

        sim.reset()
        assert sim.stats == []
        assert sim.snapshots == []