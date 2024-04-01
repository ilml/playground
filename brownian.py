# use wiener process to simulate brownian motion
# W(x + u) - W(x) ~ N(0, u)

import matplotlib.pyplot as plt
import sys
import numpy as np


def run(n: int, t: int) -> None:
    m = np.random.normal(0, 1, size=(n, t))
    m = np.cumsum(m, axis=1)

    # draw it
    n, t = m.shape

    fig, ax = plt.subplots()

    # Iterate over each row to plot
    for i in range(n):
        # Use the column indices as x and the row values as y
        x = np.arange(t)
        y = m[i, :]

        # Plot the points for this row and connect them
        ax.plot(x, y, marker='*', linestyle='-')

    # Adding labels and title for clarity
    ax.set_xlabel('time')
    ax.set_ylabel('distance')
    ax.set_title('brownian motion in 1-d')
    ax.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    n_traj, n_step = [int(x) for x in sys.argv[1:]]
    run(n_traj, n_step)

