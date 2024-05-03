# plot bezier curve

import numpy as np
import matplotlib.pyplot as plt


def bezier(n: int, cp: np.ndarray, t: np.ndarray) -> np.ndarray:
    """n: order, equal to size of of control point - 1
       cp: control points, size = n + 1
       t: points to eval the curve to draw it
    """

    if n == 0:
        return cp[0]
    else:
        return (1 - t) * bezier(n-1, cp[:n+1], t) + t * bezier(n-1, cp[1:], t)


def plot(curve: np.ndarray, cp: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(curve[:, 0], curve[:, 1], 'b-', label="Bezier Curve")  # Bezier curve
    plt.plot(cp[:, 0], cp[:, 1], 'ro-', label="Control Points")  # Control points
    plt.legend()
    plt.title('2D Bezier Curve')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # just do it within 0, 1
    t = np.linspace(0, 1, 100).reshape(-1, 1)

    # test case 1: n = 1
    cp = np.array([[0, 0], [1, 1]])

    # test case 2: n = 2
    cp = np.array([[0, 1], [0, 0], [1, 0]])

    # test case 3: n = 2
    cp = np.array([[1, 1], [0, 0], [1, 0]])

    # test case 4: n = 3
    cp = np.array([[0, 0], [0.3, 1], [0.7, 0], [1, 1]])

    # random
    cp = np.random.rand(3, 2)

    curve = bezier(cp.shape[0] - 1, cp, t)
    plot(curve, cp)

