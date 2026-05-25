import statistics
from typing import Callable

from pip._internal.resolution.resolvelib import candidates

from simulator import Simulator


class PhaseAnalyzer:
    """
    Performs phase transition analysis using Monte Carlo simulations
    to estimate the critical parameter λ_c for different system sizes.
    """

    def __init__(
        self,
        lambda_start: float,
        lambda_step: float,
        sizes: list[int],
        simulator_factory: Callable[[int], Simulator],
        crit_finder: Callable[[dict[float, float]], float],
    ):
        """
        Initialize the phase analyzer.

        Args:
            lambda_start (float): Starting value of λ.
            lambda_step (float): Step size for λ sweep.
            sizes (list[int]): List of system sizes to analyze.
            simulator_factory: Function that creates a Simulator instance for a given size.
        """
        self.lambda_start = lambda_start
        self.lambda_step = lambda_step
        self.sizes = sizes
        self.lambda_crit: float | None = None
        self.simulator = None
        self.simulator_factory = simulator_factory
        self.crit_finder = crit_finder

    def compute_final_reach(self, stiflers: list[int]) -> float:
        """
        Compute the average final number of stiflers.

        Args:
            stiflers (list[int]): Final stifler counts from multiple runs.

        Returns:
            float: Mean final reach.
        """
        return statistics.mean(stiflers)

    def find_param_crit(self, phase_results: dict) -> float:
        """
        Estimate critical parameter λ_c based on phase results.

        Args:
            phase_results (dict): Mapping from λ to measured final reach.

        Returns:
            float: Estimated critical λ (first non-zero value).
        """
        candidates = [x for x in phase_results if phase_results[x] > 0]
        if not candidates:
            raise ValueError("No non-zero values in phase results.")
        return min(candidates)

    def run(self, n_runs: int) -> dict[str, float]:
        """
        Run phase transition analysis for all system sizes.

        Args:
            n_runs (int): Number of Monte Carlo simulations per parameter value.

        Returns:
            dict: Mapping from system size to estimated λ_c.
        """
        result = {}
        for size in self.sizes:
            self.simulator = self.simulator_factory(size)
            lamb = self.lambda_start
            phase_results = {}

            while lamb < 1:
                self.simulator.set_spread_prob(lamb)
                all_runs = self.simulator.run_monte_carlo(n_runs)

                stiflers = [run[-1]["stifler"] for run in all_runs]
                phase_results[lamb] = self.compute_final_reach(stiflers)

                lamb += self.lambda_step
            if phase_results:
                self.lambda_crit = self.crit_finder(phase_results)
                result[size] = self.lambda_crit

        return result

    def find_inflection_point(self, phase_results: dict) -> float:
        """
        Estimate the inflection point of the phase transition curve.

        The inflection point is approximated by finding the value of λ
        where the change in the order parameter (final reach) is maximal.

        Args:
            phase_results (dict): Mapping from λ values to measured final reach.

        Returns:
            float: Estimated λ corresponding to the steepest increase.
        """
        lambdas = list(phase_results.keys())
        reaches = list(phase_results.values())
        diffs = [reaches[i + 1] - reaches[i] for i in range(len(reaches) - 1)]
        max_diff = diffs.index(max(diffs))
        return lambdas[max_diff]


