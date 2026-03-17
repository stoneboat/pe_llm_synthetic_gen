"""
Differential privacy accounting for the AUG-PE algorithm.

Implements Theorem 2 from Appendix A of the paper:

    The algorithm satisfies (epsilon, delta)-DP if and only if:
        Phi(sqrt(T)/(2*sigma) - epsilon*sigma/sqrt(T))
        - exp(epsilon) * Phi(-sqrt(T)/(2*sigma) - epsilon*sigma/sqrt(T))
        <= delta

    where Phi is the standard normal CDF, T is the number of iterations,
    and sigma is the noise multiplier applied to each histogram bin.

The sensitivity of each histogram is 1 (each private sample contributes
exactly one vote to one bin). Composition over T iterations yields an
effective sensitivity of sqrt(T) by the adaptive composition theorem
for Gaussian mechanisms (Corollary 3, Dong et al. 2019).
"""

import math
from scipy.stats import norm
from scipy.optimize import brentq


def _dp_condition(epsilon: float, sigma: float, T: int) -> float:
    """LHS of the DP condition from Theorem 2. Must be <= delta for DP to hold."""
    sqrt_T = math.sqrt(T)
    term1 = sqrt_T / (2 * sigma) - epsilon * sigma / sqrt_T
    term2 = -sqrt_T / (2 * sigma) - epsilon * sigma / sqrt_T
    return norm.cdf(term1) - math.exp(epsilon) * norm.cdf(term2)


def compute_epsilon(sigma: float, T: int, delta: float,
                    eps_low: float = 0.0, eps_high: float = 500.0) -> float:
    """
    Compute the privacy budget epsilon given sigma, T, and delta.

    Uses binary search (Brent's method) to find the largest epsilon such that
    the Gaussian mechanism satisfies (epsilon, delta)-DP.

    Args:
        sigma: Noise multiplier for each histogram bin.
        T: Number of PE iterations.
        delta: Target delta for (epsilon, delta)-DP.
        eps_low: Lower bound for search.
        eps_high: Upper bound for search.

    Returns:
        The epsilon such that (epsilon, delta)-DP is satisfied.
    """
    if sigma == 0:
        return math.inf

    def f(eps):
        return _dp_condition(eps, sigma, T) - delta

    if f(eps_high) > 0:
        return math.inf

    return brentq(f, eps_low, eps_high)


def compute_sigma(epsilon: float, T: int, delta: float,
                  sigma_low: float = 0.01, sigma_high: float = 1000.0) -> float:
    """
    Compute the required noise multiplier sigma for a target (epsilon, delta)-DP.

    Args:
        epsilon: Target privacy budget.
        T: Number of PE iterations.
        delta: Target delta for (epsilon, delta)-DP.
        sigma_low: Lower bound for search.
        sigma_high: Upper bound for search.

    Returns:
        The minimum sigma such that (epsilon, delta)-DP is satisfied.
    """
    if math.isinf(epsilon):
        return 0.0

    def f(sig):
        return _dp_condition(epsilon, sig, T) - delta

    return brentq(f, sigma_low, sigma_high)


def compute_delta_default(n_priv: int) -> float:
    """
    Compute the default delta following Yue et al. (2023):
        delta = 1 / (n_priv * log(n_priv))

    Args:
        n_priv: Number of private training samples.

    Returns:
        The delta value.
    """
    return 1.0 / (n_priv * math.log(n_priv))


def print_privacy_table(dataset: str = "yelp"):
    """Print the noise multipliers for standard epsilon values, matching the paper."""
    dataset_sizes = {"yelp": 1_939_290, "openreview": 8_396, "pubmed": 75_316}
    n_priv = dataset_sizes[dataset]
    delta = compute_delta_default(n_priv)
    T = 10

    print(f"Dataset: {dataset} (n={n_priv:,}, delta={delta:.2e}, T={T})")
    print(f"{'epsilon':>10} {'sigma (computed)':>18} {'sigma (paper)':>15}")
    print("-" * 48)

    paper_sigmas = {
        "yelp": {1.0: 15.34, 2.0: 8.03, 4.0: 4.24},
        "openreview": {1.0: 11.60, 2.0: 6.22, 4.0: 3.38},
        "pubmed": {1.0: 13.26, 2.0: 7.01, 4.0: 3.75},
    }

    for eps in [1.0, 2.0, 4.0]:
        sigma = compute_sigma(eps, T, delta)
        paper_sigma = paper_sigmas[dataset][eps]
        print(f"{eps:>10.1f} {sigma:>18.2f} {paper_sigma:>15.2f}")


if __name__ == "__main__":
    for ds in ["yelp", "openreview", "pubmed"]:
        print_privacy_table(ds)
        print()
