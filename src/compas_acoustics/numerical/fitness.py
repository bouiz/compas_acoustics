import math


__all__ = ['regularized_fitness']

def regularized_fitness(values: list[float], target: float, k: float = 1.0) -> float:
    """
    A loss function that combines the Mean Absolute Error (MAE) and Standard Deviation (STD)
    of a list of values with respect to a target value.
    MAE evaluates the Accuracy to the target, while STD evaluates the Consistency of the values.
    The lower the fitness score, the better the fitness.

    Parameters
    ----------
    values : list of float
        A list of numerical values to evaluate.
    target : float
        The target value to compare against.
    k : float, optional
        A regularization parameter that controls the weight of the STD in the loss function.
        Default is 1.0.
    
    Returns
    -------
    float
        The fitness score, a lower value indicates better fitness.
    """

    n = float(len(values))
    if n == 0:
        return 999.0
    
    # Mean Absolute Error (MAE)
    mae = sum(abs(t - target) for t in values) / n

    # Standard Deviation (STD)
    mean = sum(values) / n
    if n > 1:
        variance = sum((t - mean) ** 2 for t in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    return mae + k * std