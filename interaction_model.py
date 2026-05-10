from agent import State, Agent
from typing import Callable
import numpy as np

class InteractionModel:
    """
    Defines interaction rules between agents in the rumor spreading model.
    """

    def __init__(self, spread_prob: float, stifle_prob: float, cooperate_prob: float, rng: np.random.Generator) -> None:
        """
        Initialize the interaction model.

        Args:
            spread_prob (float): Probability that a spreader converts an ignorant.
            stifle_prob (float): Probability that a spreader becomes a stifler.
            cooperate_prob (float): Probability that an ignorant becomes a cooperator.
            rng (np.random.Generator): Random number generator.
        """
        self._spread_prob = spread_prob
        self._stifle_prob = stifle_prob
        self._cooperate_prob = cooperate_prob
        self.rng = rng

        self.rules: dict[tuple[State, State], Callable[[Agent, Agent], None]] = {
            (State.spreader, State.ignorant): self.spreader_ignorant,
            (State.spreader, State.spreader): self.spreader_spreader,
            (State.spreader, State.stifler): self.spreader_stifler,
            (State.spreader, State.cooperator): self.spreader_cooperator,
            (State.ignorant, State.cooperator): self.ignorant_cooperator,
        }

    @property
    def spread_prob(self) -> float:
        """
        Get the spreading probability.
        """
        return self._spread_prob

    @spread_prob.setter
    def spread_prob(self, new_spread_prob: float) -> None:
        """
        Set the spreading probability.

        Args:
            new_spread_prob (float): Probability value in [0, 1].

        Raises:
            ValueError: If the value is outside [0, 1].
        """
        if new_spread_prob < 0 or new_spread_prob > 1:
            raise ValueError("spread probability must be between 0 and 1.")
        self._spread_prob = new_spread_prob

    @property
    def stifle_prob(self) -> float:
        """
        Get the stifling probability.
        """
        return self._stifle_prob

    @stifle_prob.setter
    def stifle_prob(self, new_stifle_prob: float) -> None:
        """
        Set the stifling probability.

        Args:
            new_stifle_prob (float): Probability value in [0, 1].

        Raises:
            ValueError: If the value is outside [0, 1].
        """
        if new_stifle_prob < 0 or new_stifle_prob > 1:
            raise ValueError("stifle probability must be between 0 and 1.")
        self._stifle_prob = new_stifle_prob

    @property
    def cooperate_prob(self) -> float:
        """
        Get the cooperation probability.
        """
        return self._cooperate_prob

    @cooperate_prob.setter
    def cooperate_prob(self, new_cooperate_prob: float) -> None:
        """
        Set the cooperation probability.

        Args:
            new_cooperate_prob (float): Probability value in [0, 1].

        Raises:
            ValueError: If the value is outside [0, 1].
        """
        if new_cooperate_prob < 0 or new_cooperate_prob > 1:
            raise ValueError("cooperate probability must be between 0 and 1.")
        self._cooperate_prob = new_cooperate_prob

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
        A spreader may convert an ignorant into a spreader or cooperator.
        """
        roll = self.rng.random()
        if roll < self._spread_prob:
            ignorant.state = State.spreader
        elif roll < self._spread_prob + self._cooperate_prob:
            ignorant.state = State.cooperator

    def spreader_spreader(self, spreader: Agent, another_spreader: Agent) -> None:
        """
        A spreader may become a stifler after interacting with another spreader.
        """
        if self.rng.random() < self.stifle_prob:
            spreader.state = State.stifler

    def spreader_stifler(self, spreader: Agent, stifler: Agent) -> None:
        """
        A spreader may become a stifler after interacting with a stifler.
        """
        if self.rng.random() < self.stifle_prob:
            spreader.state = State.stifler

    def spreader_cooperator(self, spreader: Agent, cooperator: Agent) -> None:
        """
        A spreader may become a stifler after interacting with a cooperator.
        """
        if self.rng.random() < self.stifle_prob:
            spreader.state = State.stifler

    def ignorant_cooperator(self, ignorant: Agent, cooperator: Agent) -> None:
        """
        An ignorant may become a cooperator after interacting with a cooperator.
        """
        if self.rng.random() < self.cooperate_prob:
            ignorant.state = State.cooperator