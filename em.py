# em algo for estimate 2 1-d normal distribution
import numpy as np
import sys
from scipy.stats import norm

EPS = 1e-6
STEP = 1e3

def estimate(mu_1: float, std_1: float, mu_2: float, std_2: float, points: int):
    tar = [mu_1, std_1, mu_2, std_2] 
    # so to estimate distribution of the following points
    d_1 = np.random.normal(mu_1, std_1, points)
    d_2 = np.random.normal(mu_2, std_2, points)
    d = np.concatenate((d_1, d_2))

    # initial guess, index meaning is the same for tar
    est = np.random.rand(4)

    """ don't actually need this but good init make it easier to converge :)
    est[0] = 0.5
    est[2] = 9
    est[1] = est[3] = 0.7
    """

    prev_est = np.zeros(4)
    cnt = 0

    while not all((abs(x-y) < EPS for x, y in zip(est, prev_est))) and cnt < STEP:

        prob_1 = np.array([norm.pdf(x, loc=est[0], scale=est[1]) for x in d])
        prob_2 = np.array([norm.pdf(x, loc=est[2], scale=est[3]) for x in d])

        # notice that this keep p(z|x, theta) a probability 
        total = prob_1 + prob_2
        prob_1 /= total
        prob_2 /= total

        prob_1_s = np.sum(prob_1)
        prob_2_s = np.sum(prob_2)


        # save est for condition checking
        prev_est = np.copy(est)
        # estimate mean

        est[0] = np.sum(prob_1 * d) / prob_1_s
        est[2] = np.sum(prob_2 * d) / prob_2_s

        # estimate std dev
        est[1] = np.sqrt(np.sum(prob_1 * (d - est[0])**2) / prob_1_s)
        est[3] = np.sqrt(np.sum(prob_2 * (d - est[2])**2) / prob_2_s)

        cnt += 1
        print(f'iter: {cnt} | tar: {tar} |  estimate: {est}')



if __name__ == "__main__":
    mu_1, std_1, mu_2, std_2 = [float(x) for x in sys.argv[1:5]]
    points = int(sys.argv[5])
    estimate(mu_1, std_1, mu_2, std_2, points)