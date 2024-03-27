import numpy as np
from typing import Any
import sys


def norm_dist_pdf(x: float, mean: float=0, std_dev: float=1) -> float:
    factor = 1 / (std_dev * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mean) / std_dev)**2
    return factor * np.exp(exponent)


def uniform_pdf(x: float) -> float:
    return 1


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
    

def m_c(mean: float, std_var: float, times: int) -> float:
    """ use mc method to estimate mean of a norm dist
    """
    sample = []
    for _ in range(times):
        x = uniform01()
        sample += [x * i_s(norm_dist_pdf, uniform_pdf, x)]
    estimate = np.mean(sample)
    return estimate 


if __name__ == "__main__":
    m_c(sys.argv[1])



