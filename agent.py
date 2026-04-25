from enum import Enum, auto

class State(Enum):
    ignorant = auto()
    spreader = auto()
    stifler = auto()


class Agent:
    def __init__(self, state, position):
        self.state = state
        self._position = position

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if not isinstance(value, State):
            raise TypeError('State must be an instance of State')
        else:
            self._state = value
