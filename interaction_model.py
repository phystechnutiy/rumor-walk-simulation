from agent import State, Agent
import numpy as np

class InteractionModel:
    """
    Defines interaction rules between agents in the rumor spreading model.
    """

    def __init__(self, _lambda: float, _alpha: float, rng: np.random.Generator) -> None:
        """
        Initialize the interaction model.

        Args:
            _lambda (float): Probability that a spreader converts an ignorant.
            _alpha (float): Probability that a spreader becomes a stifler.
            rng (np.random.Generator): Random number generator.
        """
        self._lambda = _lambda
        self._alpha = _alpha
        self.rng = rng

        self.rules: dict[tuple[State, State], callable] = {
            (State.spreader, State.ignorant): self.spreader_ignorant,
            (State.spreader, State.spreader): self.spreader_spreader,
            (State.spreader, State.stifler): self.spreader_stifler,
        }

    def interact(self, a: Agent, b: Agent) -> None:
        """
        Apply interaction rules between two agents if applicable.
        """
        key = (a.state, b.state)
        if key in self.rules:
            self.rules[key](a, b)
        else:
            key = (b.state, a.state)
            if key in self.rules:
                self.rules[key](b, a)

    def spreader_ignorant(self, spreader: Agent, ignorant: Agent) -> None:
        """
        A spreader may convert an ignorant into a spreader.
        """
        if self.rng.random() < self._lambda:
            ignorant.state = State.spreader

    def spreader_spreader(self, spreader: Agent, another_spreader: Agent) -> None:
        """
        A spreader may become a stifler after interacting with another spreader.
        """
        if self.rng.random() < self._alpha:
            spreader.state = State.stifler

    def spreader_stifler(self, spreader: Agent, stifler: Agent) -> None:
        """
        A spreader may become a stifler after interacting with a stifler.
        """
        if self.rng.random() < self._alpha:
            spreader.state = State.stifler