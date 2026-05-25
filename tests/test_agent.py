import pytest
from agent import Agent, State


class TestState:
    def test_all_four_states_exist(self):
        assert State.ignorant
        assert State.spreader
        assert State.stifler
        assert State.cooperator

    def test_states_are_unique(self):
        values = [s.value for s in State]
        assert len(values) == len(set(values))

    def test_state_is_enum(self):
        for s in State:
            assert isinstance(s, State)


class TestAgentInit:
    def test_init_ignorant(self):
        agent = Agent(State.ignorant, 0)
        assert agent.state == State.ignorant
        assert agent.position == 0

    def test_init_spreader(self):
        agent = Agent(State.spreader, 5)
        assert agent.state == State.spreader
        assert agent.position == 5

    def test_init_stifler(self):
        agent = Agent(State.stifler, 10)
        assert agent.state == State.stifler
        assert agent.position == 10

    def test_init_cooperator(self):
        agent = Agent(State.cooperator, 3)
        assert agent.state == State.cooperator
        assert agent.position == 3

    def test_init_invalid_state_raises_type_error(self):
        with pytest.raises(TypeError):
            Agent("ignorant", 0)

    def test_init_invalid_state_int_raises_type_error(self):
        with pytest.raises(TypeError):
            Agent(0, 0)

    def test_init_invalid_state_none_raises_type_error(self):
        with pytest.raises(TypeError):
            Agent(None, 0)


class TestAgentStateSetter:
    def test_set_state_to_spreader(self):
        agent = Agent(State.ignorant, 0)
        agent.state = State.spreader
        assert agent.state == State.spreader

    def test_set_state_to_stifler(self):
        agent = Agent(State.spreader, 0)
        agent.state = State.stifler
        assert agent.state == State.stifler

    def test_set_state_to_cooperator(self):
        agent = Agent(State.ignorant, 0)
        agent.state = State.cooperator
        assert agent.state == State.cooperator

    def test_set_state_invalid_string_raises_type_error(self):
        agent = Agent(State.ignorant, 0)
        with pytest.raises(TypeError):
            agent.state = "spreader"

    def test_set_state_invalid_int_raises_type_error(self):
        agent = Agent(State.ignorant, 0)
        with pytest.raises(TypeError):
            agent.state = 1

    def test_set_state_invalid_none_raises_type_error(self):
        agent = Agent(State.ignorant, 0)
        with pytest.raises(TypeError):
            agent.state = None

    def test_type_error_message(self):
        agent = Agent(State.ignorant, 0)
        with pytest.raises(TypeError, match="State must be an instance of State"):
            agent.state = "invalid"


class TestAgentPositionSetter:
    def test_set_position_valid(self):
        agent = Agent(State.ignorant, 0)
        agent.position = 7
        assert agent.position == 7

    def test_set_position_zero(self):
        agent = Agent(State.ignorant, 5)
        agent.position = 0
        assert agent.position == 0

    def test_set_position_large(self):
        agent = Agent(State.ignorant, 0)
        agent.position = 999
        assert agent.position == 999

    def test_set_position_float_raises_type_error(self):
        agent = Agent(State.ignorant, 0)
        with pytest.raises(TypeError):
            agent.position = 1.5

    def test_set_position_string_raises_type_error(self):
        agent = Agent(State.ignorant, 0)
        with pytest.raises(TypeError):
            agent.position = "3"

    def test_set_position_none_raises_type_error(self):
        agent = Agent(State.ignorant, 0)
        with pytest.raises(TypeError):
            agent.position = None

    def test_type_error_message(self):
        agent = Agent(State.ignorant, 0)
        with pytest.raises(TypeError, match="Position must be an integer"):
            agent.position = 3.14