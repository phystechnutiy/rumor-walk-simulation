import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from agent import Agent, State
from graph import ErdosRenyiGraph
from simulator import Simulator
from interaction_model import InteractionModel


def make_rng(seed=42):
    return np.random.default_rng(seed)

class TestBuildSimulator:
    def _build(self, n=5, p=1.0, spread=0.5, stifle=0.3, cooperate=0.1, seed=42):
        from main import build_simulator
        rng = make_rng(seed)
        return build_simulator(n, rng, p, spread, stifle, cooperate)

    def test_returns_tuple(self):
        result = self._build()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_simulator(self):
        sim, _ = self._build()
        assert isinstance(sim, Simulator)

    def test_second_element_is_erdos_renyi_graph(self):
        _, graph = self._build()
        assert isinstance(graph, ErdosRenyiGraph)

    def test_correct_number_of_agents(self):
        sim, _ = self._build(n=7)
        assert len(sim.agents) == 7

    def test_first_agent_is_spreader(self):
        sim, _ = self._build()
        assert sim.agents[0].state == State.spreader

    def test_remaining_agents_are_ignorant(self):
        sim, _ = self._build(n=5)
        for agent in sim.agents[1:]:
            assert agent.state == State.ignorant

    def test_all_agents_have_valid_positions(self):
        n = 5
        sim, graph = self._build(n=n)
        for agent in sim.agents:
            assert agent.position in graph.node_neighbors

    def test_node_occupants_match_agent_positions(self):
        sim, _ = self._build(n=5)
        for agent in sim.agents:
            assert agent in sim.node_occupants[agent.position]

    def test_interaction_model_has_correct_probs(self):
        sim, _ = self._build(spread=0.4, stifle=0.2, cooperate=0.15)
        model = sim.interaction_model
        assert model.spread_prob == 0.4
        assert model.stifle_prob == 0.2
        assert model.cooperate_prob == 0.15


class TestMonteCarlo:
    def _make_sim_graph(self):
        from main import build_simulator
        rng = make_rng()
        return build_simulator(3, rng, 1.0, 0.5, 0.3, 0.1)

    def test_monte_carlo_prints_results(self, capsys):
        from main import monte_carlo
        rng = make_rng()
        monte_carlo(n=3, n_runs=2, rng=rng, p=1.0,
                    spread_prob=0.9, stifle_prob=0.3, cooperate_prob=0.1,
                    dashboard=False)
        captured = capsys.readouterr()
        assert "Monte Carlo results" in captured.out

    def test_monte_carlo_prints_each_run(self, capsys):
        from main import monte_carlo
        rng = make_rng()
        n_runs = 3
        monte_carlo(n=3, n_runs=n_runs, rng=rng, p=1.0,
                    spread_prob=0.9, stifle_prob=0.3, cooperate_prob=0.1,
                    dashboard=False)
        captured = capsys.readouterr()
        for i in range(n_runs):
            assert f"Run {i}" in captured.out

    def test_monte_carlo_dashboard_mode_calls_run(self):
        from main import monte_carlo
        mock_sim = MagicMock()
        mock_sim.run.return_value = None
        mock_sim.snapshots = []
        mock_graph = MagicMock()

        mock_dashboard_instance = MagicMock()

        with patch("main.build_simulator", return_value=(mock_sim, mock_graph)), \
             patch("main.SimulationDashboard", return_value=mock_dashboard_instance):
            from main import monte_carlo
            monte_carlo(n=3, n_runs=2, rng=make_rng(), p=1.0,
                        spread_prob=0.5, stifle_prob=0.3, cooperate_prob=0.1,
                        dashboard=True)
            mock_sim.run.assert_called_once()
            mock_dashboard_instance.run.assert_called_once()

    def test_monte_carlo_no_dashboard_calls_run_monte_carlo(self):
        mock_sim = MagicMock()
        mock_sim.run_monte_carlo.return_value = [
            [{"tick": 0, "ignorant": 1, "spreader": 0, "stifler": 2}]
        ]
        mock_graph = MagicMock()

        with patch("main.build_simulator", return_value=(mock_sim, mock_graph)):
            from main import monte_carlo
            monte_carlo(n=3, n_runs=1, rng=make_rng(), p=1.0,
                        spread_prob=0.5, stifle_prob=0.3, cooperate_prob=0.1,
                        dashboard=False)
            mock_sim.run_monte_carlo.assert_called_once_with(1)

class TestInitPhaseAnalyzer:
    def test_returns_phase_analyzer(self):
        from main import init_phase_analyzer
        from phase_analyzer import PhaseAnalyzer
        rng = make_rng()
        crit = MagicMock()
        pa = init_phase_analyzer(rng, 1.0, 0.5, 0.3, 0.1, 0.1, 0.2, [10, 20], crit)
        assert isinstance(pa, PhaseAnalyzer)

    def test_lambda_start_set_correctly(self):
        from main import init_phase_analyzer
        rng = make_rng()
        crit = MagicMock()
        pa = init_phase_analyzer(rng, 1.0, 0.5, 0.3, 0.1, 0.3, 0.1, [10], crit)
        assert pa.lambda_start == 0.3

    def test_lambda_step_set_correctly(self):
        from main import init_phase_analyzer
        rng = make_rng()
        crit = MagicMock()
        pa = init_phase_analyzer(rng, 1.0, 0.5, 0.3, 0.1, 0.1, 0.15, [10], crit)
        assert pa.lambda_step == 0.15

    def test_sizes_set_correctly(self):
        from main import init_phase_analyzer
        rng = make_rng()
        crit = MagicMock()
        pa = init_phase_analyzer(rng, 1.0, 0.5, 0.3, 0.1, 0.1, 0.1, [5, 15, 30], crit)
        assert pa.sizes == [5, 15, 30]

    def test_simulator_factory_returns_simulator(self):
        from main import init_phase_analyzer
        rng = make_rng()
        crit = MagicMock()
        pa = init_phase_analyzer(rng, 1.0, 0.5, 0.3, 0.1, 0.1, 0.1, [5], crit)
        sim = pa.simulator_factory(5)
        assert isinstance(sim, Simulator)

    def test_crit_finder_stored(self):
        from main import init_phase_analyzer
        rng = make_rng()
        crit = MagicMock()
        pa = init_phase_analyzer(rng, 1.0, 0.5, 0.3, 0.1, 0.1, 0.1, [5], crit)
        assert pa.crit_finder is crit


class TestCriticalLambdas:
    def test_critical_lambdas_prints_results(self, capsys):
        mock_analyzer = MagicMock()
        mock_analyzer.find_param_crit = MagicMock()
        mock_analyzer.run.return_value = {10: 0.3, 20: 0.25}

        with patch("main.init_phase_analyzer", return_value=mock_analyzer):
            from main import critical_lambdas
            critical_lambdas(
                n_runs=2, rng=make_rng(), p=1.0,
                spread_prob=0.5, stifle_prob=0.3, cooperate_prob=0.1,
                param_start=0.1, param_step=0.1, sizes=[10, 20],
                crit_finder=MagicMock()
            )
        captured = capsys.readouterr()
        assert "10" in captured.out or "0.3" in captured.out

    def test_critical_lambdas_uses_find_param_crit(self):
        mock_analyzer = MagicMock()
        mock_analyzer.find_param_crit = MagicMock(return_value=0.3)
        mock_analyzer.run.return_value = {10: 0.3}

        with patch("main.init_phase_analyzer", return_value=mock_analyzer):
            from main import critical_lambdas
            critical_lambdas(
                n_runs=2, rng=make_rng(), p=1.0,
                spread_prob=0.5, stifle_prob=0.3, cooperate_prob=0.1,
                param_start=0.1, param_step=0.1, sizes=[10],
                crit_finder=MagicMock()
            )
        assert mock_analyzer.crit_finder == mock_analyzer.find_param_crit

class TestInflectionPoints:
    def test_inflection_points_prints_results(self, capsys):
        mock_analyzer = MagicMock()
        mock_analyzer.find_inflection_point = MagicMock()
        mock_analyzer.run.return_value = {10: 0.4}

        with patch("main.init_phase_analyzer", return_value=mock_analyzer):
            from main import inflection_points
            inflection_points(
                n_runs=2, rng=make_rng(), p=1.0,
                spread_prob=0.5, stifle_prob=0.3, cooperate_prob=0.1,
                param_start=0.1, param_step=0.1, sizes=[10],
                crit_finder=MagicMock()
            )
        captured = capsys.readouterr()
        assert "10" in captured.out or "0.4" in captured.out

    def test_inflection_points_uses_find_inflection_point(self):
        mock_analyzer = MagicMock()
        mock_analyzer.find_inflection_point = MagicMock(return_value=0.4)
        mock_analyzer.run.return_value = {10: 0.4}

        with patch("main.init_phase_analyzer", return_value=mock_analyzer):
            from main import inflection_points
            inflection_points(
                n_runs=2, rng=make_rng(), p=1.0,
                spread_prob=0.5, stifle_prob=0.3, cooperate_prob=0.1,
                param_start=0.1, param_step=0.1, sizes=[10],
                crit_finder=MagicMock()
            )
        assert mock_analyzer.crit_finder == mock_analyzer.find_inflection_point