import math

__all__ = ['regularized_fitness']


def regularized_fitness(values: list[float], target: float, k: float = 1.0) -> float:
    """Mean absolute error to ``target`` plus ``k`` times the sample standard deviation; lower is better.

    Parameters
    ----------
    values : list of float
    target : float
    k : float, optional
        Weight of the standard deviation.

    Returns
    -------
    float
        999 for no values.
    """
    n = float(len(values))
    if n == 0:
        return 999.0
    mae = sum(abs(t - target) for t in values) / n
    mean = sum(values) / n
    std = math.sqrt(sum((t - mean) ** 2 for t in values) / (n - 1)) if n > 1 else 0.0
    return mae + k * std
