import pytest
import numpy as np
from unittest.mock import MagicMock
from agent import Agent, State
from interaction_model import InteractionModel


def make_model(spread=0.5, stifle=0.3, cooperate=0.2, seed=42):
    """Хелпер: создаёт InteractionModel с реальным rng."""
    rng = np.random.default_rng(seed)
    return InteractionModel(spread, stifle, cooperate, rng)


def make_model_with_mock_rng(return_value: float):
    """Хелпер: создаёт InteractionModel с rng, который всегда возвращает return_value."""
    rng = MagicMock()
    rng.random.return_value = return_value
    return InteractionModel(0.5, 0.3, 0.2, rng)


def agent(state, pos=0):
    return Agent(state, pos)


class TestInit:
    def test_spread_prob_stored(self):
        m = make_model(spread=0.4)
        assert m.spread_prob == 0.4

    def test_stifle_prob_stored(self):
        m = make_model(stifle=0.3)
        assert m.stifle_prob == 0.3

    def test_cooperate_prob_stored(self):
        m = make_model(cooperate=0.2)
        assert m.cooperate_prob == 0.2

    def test_rules_dict_has_five_entries(self):
        m = make_model()
        assert len(m.rules) == 5

    def test_rules_contains_expected_keys(self):
        m = make_model()
        expected_keys = {
            (State.spreader, State.ignorant),
            (State.spreader, State.spreader),
            (State.spreader, State.stifler),
            (State.spreader, State.cooperator),
            (State.ignorant, State.cooperator),
        }
        assert set(m.rules.keys()) == expected_keys



class TestSpreadProbSetter:
    def test_set_valid_value(self):
        m = make_model()
        m.spread_prob = 0.9
        assert m.spread_prob == 0.9

    def test_set_zero(self):
        m = make_model()
        m.spread_prob = 0.0
        assert m.spread_prob == 0.0

    def test_set_one(self):
        m = make_model()
        m.spread_prob = 1.0
        assert m.spread_prob == 1.0

    def test_set_negative_raises_value_error(self):
        m = make_model()
        with pytest.raises(ValueError):
            m.spread_prob = -0.1

    def test_set_above_one_raises_value_error(self):
        m = make_model()
        with pytest.raises(ValueError):
            m.spread_prob = 1.1


class TestStiffleProbSetter:
    def test_set_valid_value(self):
        m = make_model()
        m.stifle_prob = 0.5
        assert m.stifle_prob == 0.5

    def test_set_negative_raises_value_error(self):
        m = make_model()
        with pytest.raises(ValueError):
            m.stifle_prob = -0.01

    def test_set_above_one_raises_value_error(self):
        m = make_model()
        with pytest.raises(ValueError):
            m.stifle_prob = 1.5


class TestCooperateProbSetter:
    def test_set_valid_value(self):
        m = make_model()
        m.cooperate_prob = 0.3
        assert m.cooperate_prob == 0.3

    def test_set_negative_raises_value_error(self):
        m = make_model()
        with pytest.raises(ValueError):
            m.cooperate_prob = -1.0

    def test_set_above_one_raises_value_error(self):
        m = make_model()
        with pytest.raises(ValueError):
            m.cooperate_prob = 2.0



class TestSpreaderIgnorant:
    def test_ignorant_becomes_spreader_when_roll_below_spread_prob(self):
        m = make_model_with_mock_rng(0.1)
        spreader = agent(State.spreader)
        ignorant = agent(State.ignorant)
        m.spreader_ignorant(spreader, ignorant)
        assert ignorant.state == State.spreader

    def test_ignorant_becomes_cooperator_when_roll_in_cooperate_band(self):
        m = make_model_with_mock_rng(0.6)
        spreader = agent(State.spreader)
        ignorant = agent(State.ignorant)
        m.spreader_ignorant(spreader, ignorant)
        assert ignorant.state == State.cooperator

    def test_ignorant_stays_ignorant_when_roll_above_both_probs(self):
        m = make_model_with_mock_rng(0.9)
        spreader = agent(State.spreader)
        ignorant = agent(State.ignorant)
        m.spreader_ignorant(spreader, ignorant)
        assert ignorant.state == State.ignorant

    def test_spreader_state_unchanged(self):
        m = make_model_with_mock_rng(0.1)
        spreader = agent(State.spreader)
        ignorant = agent(State.ignorant)
        m.spreader_ignorant(spreader, ignorant)
        assert spreader.state == State.spreader


class TestSpreaderSpreader:
    def test_spreader_becomes_stifler_when_roll_below_stifle_prob(self):
        m = make_model_with_mock_rng(0.1)
        s1 = agent(State.spreader)
        s2 = agent(State.spreader)
        m.spreader_spreader(s1, s2)
        assert s1.state == State.stifler

    def test_spreader_stays_spreader_when_roll_above_stifle_prob(self):
        m = make_model_with_mock_rng(0.8)
        s1 = agent(State.spreader)
        s2 = agent(State.spreader)
        m.spreader_spreader(s1, s2)
        assert s1.state == State.spreader

    def test_second_spreader_state_unchanged(self):
        m = make_model_with_mock_rng(0.1)
        s1 = agent(State.spreader)
        s2 = agent(State.spreader)
        m.spreader_spreader(s1, s2)
        assert s2.state == State.spreader


class TestSpreaderStifler:
    def test_spreader_becomes_stifler_when_roll_below_stifle_prob(self):
        m = make_model_with_mock_rng(0.1)
        spreader = agent(State.spreader)
        stifler = agent(State.stifler)
        m.spreader_stifler(spreader, stifler)
        assert spreader.state == State.stifler

    def test_spreader_stays_when_roll_above_stifle_prob(self):
        m = make_model_with_mock_rng(0.9)
        spreader = agent(State.spreader)
        stifler = agent(State.stifler)
        m.spreader_stifler(spreader, stifler)
        assert spreader.state == State.spreader

    def test_stifler_state_unchanged(self):
        m = make_model_with_mock_rng(0.1)
        spreader = agent(State.spreader)
        stifler = agent(State.stifler)
        m.spreader_stifler(spreader, stifler)
        assert stifler.state == State.stifler


class TestSpreaderCooperator:
    def test_spreader_becomes_stifler_when_roll_below_stifle_prob(self):
        m = make_model_with_mock_rng(0.1)
        spreader = agent(State.spreader)
        cooperator = agent(State.cooperator)
        m.spreader_cooperator(spreader, cooperator)
        assert spreader.state == State.stifler

    def test_spreader_stays_when_roll_above_stifle_prob(self):
        m = make_model_with_mock_rng(0.9)
        spreader = agent(State.spreader)
        cooperator = agent(State.cooperator)
        m.spreader_cooperator(spreader, cooperator)
        assert spreader.state == State.spreader

    def test_cooperator_state_unchanged(self):
        m = make_model_with_mock_rng(0.1)
        spreader = agent(State.spreader)
        cooperator = agent(State.cooperator)
        m.spreader_cooperator(spreader, cooperator)
        assert cooperator.state == State.cooperator


class TestIgnorantCooperator:
    def test_ignorant_becomes_cooperator_when_roll_below_cooperate_prob(self):
        m = make_model_with_mock_rng(0.1)
        ignorant = agent(State.ignorant)
        cooperator = agent(State.cooperator)
        m.ignorant_cooperator(ignorant, cooperator)
        assert ignorant.state == State.cooperator

    def test_ignorant_stays_when_roll_above_cooperate_prob(self):
        # cooperate_prob=0.2, roll=0.5 → остаётся ignorant
        m = make_model_with_mock_rng(0.5)
        ignorant = agent(State.ignorant)
        cooperator = agent(State.cooperator)
        m.ignorant_cooperator(ignorant, cooperator)
        assert ignorant.state == State.ignorant

    def test_cooperator_state_unchanged(self):
        m = make_model_with_mock_rng(0.1)
        ignorant = agent(State.ignorant)
        cooperator = agent(State.cooperator)
        m.ignorant_cooperator(ignorant, cooperator)
        assert cooperator.state == State.cooperator


class TestInteract:
    def test_interact_spreader_ignorant_direct_order(self):
        m = make_model_with_mock_rng(0.1)
        spreader = agent(State.spreader)
        ignorant = agent(State.ignorant)
        m.interact(spreader, ignorant)
        assert ignorant.state == State.spreader

    def test_interact_ignorant_spreader_reversed_order(self):
        m = make_model_with_mock_rng(0.1)
        spreader = agent(State.spreader)
        ignorant = agent(State.ignorant)
        m.interact(ignorant, spreader)
        assert ignorant.state == State.spreader

    def test_interact_no_rule_does_nothing(self):
        m = make_model_with_mock_rng(0.1)
        s1 = agent(State.stifler)
        s2 = agent(State.stifler)
        m.interact(s1, s2)
        assert s1.state == State.stifler
        assert s2.state == State.stifler

    def test_interact_ignorant_cooperator_direct(self):
        m = make_model_with_mock_rng(0.1)
        ignorant = agent(State.ignorant)
        cooperator = agent(State.cooperator)
        m.interact(ignorant, cooperator)
        assert ignorant.state == State.cooperator

    def test_interact_cooperator_ignorant_reversed(self):
        m = make_model_with_mock_rng(0.1)
        ignorant = agent(State.ignorant)
        cooperator = agent(State.cooperator)
        m.interact(cooperator, ignorant)
        assert ignorant.state == State.cooperator