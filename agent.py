from enum import Enum, auto

class State(Enum):
    """Represents the possible states of an agent in the rumor spreading model."""
    ignorant = auto()   
    spreader = auto()   
    stifler = auto()  


class Agent:
    """Represents an individual agent in the rumor spreading simulation."""

    def __init__(self, state: State, position: int) -> None:
        """
        Initialize an agent.

        Args:
            state (State): Initial state of the agent.
            position (int): Initial position of the agent.
        """
        self.state = state
        self._position = position

    @property
    def state(self) -> State:
        """Get the current state of the agent."""
        return self._state

    @state.setter
    def state(self, value: State) -> None:
        """
        Set the state of the agent.

        Raises:
            TypeError: If value is not an instance of State.
        """
        if not isinstance(value, State):
            raise TypeError('State must be an instance of State')
        else:
            self._state = value

    @property
    def position(self) -> int:
        """Get the current position of the agent."""
        return self._position

    @position.setter
    def position(self, value: int) -> None:
        """
        Set the position of the agent.

        Raises:
            TypeError: If value is not an integer.
        """
        if not isinstance(value, int):
            raise TypeError('Position must be an integer')
        else:
            self._position = value
