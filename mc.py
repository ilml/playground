import numpy as np
from typing import Any
import sys
from functools import partial


def norm_dist_prob(x: float, mean: float=0, std_dev: float=1) -> float:
    factor = 1 / (std_dev * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mean) / std_dev)**2
    return factor * np.exp(exponent)


def uniform(a: float, b: float) -> float:
    """uniformly generate a number within a and b, b >= a
    """
    return  a + (b - a) * uniform01()


def uniform_prob(x: float, a: float, b: float) -> float:
    """of course x is not used here
    """
    return 1 / (b - a)


def uniform01() -> float:
    return np.random.uniform(0, 1)


def i_s(f: Any, g: Any, x: float) -> np.ndarray:
    """importance sampling

    Args:
        f: desired distribution
        g: proposed distribution
        x: sample
    """
    weight = f(x) / g(x)
    return weight
    

def m_c(mean: float, std_dev: float, a: float, b: float, times: int) -> float:
    """ use mc method to estimate mean of a norm dist
    """
    norm_dist_p = partial(norm_dist_prob, mean=mean, std_dev=std_dev)
    uniform_prob_p = partial(uniform_prob, a=a, b=b)
    samples = np.array([uniform(a, b) for _ in range(times)])
    weights = np.array([i_s(norm_dist_p, uniform_prob_p, x) for x in samples])
    estimate = np.sum(samples * weights) / np.sum(weights)

    return estimate 


if __name__ == "__main__":
    mean, std_dev, a, b, times = sys.argv[1:]
    mean = float(mean)
    std_dev = float(std_dev)
    times = int(times)
    a = float(a)
    b = float(b)
    estimate_mean = m_c(mean, std_dev, a, b, times)
    print(f'real mean: {mean}, estimate mean: {estimate_mean}')



