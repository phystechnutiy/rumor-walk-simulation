import pytest
from unittest.mock import MagicMock, patch
from phase_analyzer import PhaseAnalyzer


def make_mock_simulator(stifler_per_run=5):
    """
    Создаёт мок-симулятор: run_monte_carlo возвращает n_runs прогонов,
    каждый из которых — список из одного тика с заданным числом stifler'ов.
    """
    sim = MagicMock()
    sim.run_monte_carlo.side_effect = lambda n_runs: [
        [{"tick": 0, "ignorant": 0, "spreader": 0, "stifler": stifler_per_run}]
        for _ in range(n_runs)
    ]
    return sim


def make_analyzer(
    lambda_start=0.1,
    lambda_step=0.3,
    sizes=None,
    stifler_per_run=5,
    crit_finder=None,
):
    if sizes is None:
        sizes = [10]
    sim = make_mock_simulator(stifler_per_run)
    factory = MagicMock(return_value=sim)
    if crit_finder is None:
        crit_finder = PhaseAnalyzer(0, 1, [], factory, lambda x: 0).find_param_crit
    return PhaseAnalyzer(lambda_start, lambda_step, sizes, factory, crit_finder), factory, sim

class TestInit:
    def test_lambda_start_stored(self):
        a, _, _ = make_analyzer(lambda_start=0.2)
        assert a.lambda_start == 0.2

    def test_lambda_step_stored(self):
        a, _, _ = make_analyzer(lambda_step=0.1)
        assert a.lambda_step == 0.1

    def test_sizes_stored(self):
        a, _, _ = make_analyzer(sizes=[10, 50])
        assert a.sizes == [10, 50]

    def test_lambda_crit_is_none_on_init(self):
        a, _, _ = make_analyzer()
        assert a.lambda_crit is None

    def test_simulator_is_none_on_init(self):
        a, _, _ = make_analyzer()
        assert a.simulator is None


class TestComputeFinalReach:
    def test_mean_of_equal_values(self):
        a, _, _ = make_analyzer()
        assert a.compute_final_reach([4, 4, 4]) == 4.0

    def test_mean_of_mixed_values(self):
        a, _, _ = make_analyzer()
        assert a.compute_final_reach([2, 4, 6]) == 4.0

    def test_single_value(self):
        a, _, _ = make_analyzer()
        assert a.compute_final_reach([7]) == 7.0

    def test_zeros(self):
        a, _, _ = make_analyzer()
        assert a.compute_final_reach([0, 0, 0]) == 0.0

    def test_returns_float(self):
        a, _, _ = make_analyzer()
        result = a.compute_final_reach([1.0, 2.0, 3.0])
        assert isinstance(result, float)


class TestFindParamCrit:
    def make_bare(self):
        factory = MagicMock()
        crit = MagicMock()
        return PhaseAnalyzer(0.1, 0.1, [10], factory, crit)

    def test_returns_first_nonzero_lambda(self):
        a = self.make_bare()
        phase = {0.1: 0.0, 0.2: 0.0, 0.3: 3.0, 0.4: 5.0}
        assert a.find_param_crit(phase) == 0.3

    def test_returns_minimum_nonzero_key(self):
        a = self.make_bare()
        phase = {0.5: 2.0, 0.2: 1.0, 0.8: 3.0}
        assert a.find_param_crit(phase) == 0.2

    def test_all_zeros_raises_value_error(self):
        a = self.make_bare()
        phase = {0.1: 0.0, 0.2: 0.0}
        with pytest.raises(ValueError):
            a.find_param_crit(phase)

    def test_single_nonzero(self):
        a = self.make_bare()
        phase = {0.1: 0.0, 0.5: 3.5}
        assert a.find_param_crit(phase) == 0.5

    def test_all_nonzero_returns_minimum(self):
        a = self.make_bare()
        phase = {0.3: 1.0, 0.6: 2.0, 0.9: 3.0}
        assert a.find_param_crit(phase) == 0.3


class TestFindInflectionPoint:
    def make_bare(self):
        factory = MagicMock()
        crit = MagicMock()
        return PhaseAnalyzer(0.1, 0.1, [10], factory, crit)

    def test_finds_lambda_with_max_diff(self):
        a = self.make_bare()
        phase = {0.1: 0.0, 0.2: 1.0, 0.3: 11.0, 0.4: 12.0}
        assert a.find_inflection_point(phase) == 0.2

    def test_first_step_is_steepest(self):
        a = self.make_bare()
        phase = {0.1: 0.0, 0.2: 10.0, 0.3: 11.0, 0.4: 11.5}
        assert a.find_inflection_point(phase) == 0.1

    def test_last_step_is_steepest(self):
        a = self.make_bare()
        phase = {0.1: 0.0, 0.2: 1.0, 0.3: 2.0, 0.4: 20.0}
        assert a.find_inflection_point(phase) == 0.3

    def test_returns_key_not_value(self):
        a = self.make_bare()
        phase = {0.2: 0.0, 0.5: 100.0}
        result = a.find_inflection_point(phase)
        assert result in phase.keys()


class TestRun:
    def test_run_returns_dict(self):
        a, _, _ = make_analyzer(sizes=[10])
        result = a.run(n_runs=2)
        assert isinstance(result, dict)

    def test_run_keys_are_sizes(self):
        a, _, _ = make_analyzer(sizes=[10, 20])
        result = a.run(n_runs=2)
        assert set(result.keys()) == {10, 20}

    def test_run_calls_factory_for_each_size(self):
        a, factory, _ = make_analyzer(sizes=[10, 20, 50])
        a.run(n_runs=2)
        assert factory.call_count == 3

    def test_run_calls_factory_with_correct_size(self):
        a, factory, _ = make_analyzer(sizes=[42])
        a.run(n_runs=2)
        factory.assert_called_with(42)

    def test_run_calls_monte_carlo_for_each_lambda(self):
        a, _, sim = make_analyzer(lambda_start=0.1, lambda_step=0.3, sizes=[10])
        a.run(n_runs=2)
        assert sim.run_monte_carlo.call_count == 3

    def test_run_calls_monte_carlo_with_correct_n_runs(self):
        a, _, sim = make_analyzer(lambda_start=0.5, lambda_step=0.3, sizes=[10])
        a.run(n_runs=5)
        for call in sim.run_monte_carlo.call_args_list:
            assert call.args[0] == 5

    def test_run_stores_lambda_crit(self):
        a, _, _ = make_analyzer(sizes=[10], stifler_per_run=5)
        a.run(n_runs=2)
        assert a.lambda_crit is not None

    def test_run_empty_sizes_returns_empty_dict(self):
        factory = MagicMock()
        crit = MagicMock()
        a = PhaseAnalyzer(0.1, 0.3, [], factory, crit)
        result = a.run(n_runs=2)
        assert result == {}
        factory.assert_not_called()

    def test_run_uses_crit_finder(self):
        custom_crit = MagicMock(return_value=0.42)
        factory = MagicMock(return_value=make_mock_simulator(stifler_per_run=3))
        a = PhaseAnalyzer(0.1, 0.3, [10], factory, custom_crit)
        result = a.run(n_runs=2)
        custom_crit.assert_called()
        assert result[10] == 0.42