# try ways to optimise matmul on cpu 

import time
import numpy as np
import sys


def numpy_api(M: np.ndarray, N: np.ndarray) -> np.ndarray:
    return np.matmul(M, N)
    """numpy """

def matmul_1(M: np.ndarray, N : np.ndarray) -> np.ndarray:
    """vallina"""
    ii, jj = M.shape
    _, kk = N.shape
    C = np.zeros((ii, kk))
    for i in range(ii):
        for k in range(kk):
            for j in range(jj):
                C[i, k] += M[i, j] * N[j, k]
    return C

    
def matmul_2(M: np.ndarray, N : np.ndarray) -> np.ndarray:
    """reorder"""
    ii, jj = M.shape
    _, kk = N.shape
    C = np.zeros((ii, kk))
    for i in range(ii):
        for j in range(jj):
            for k in range(kk):
                C[i, k] += M[i, j] * N[j, k]
    return C



if __name__ == "__main__":
    # tiny unit test
    """
    M = np.array([[1, 2], [3, 4]])
    N = np.array([[2, 3], [4, 5]])
    print(matmul_1(M, N)) # [[8, 5], [20, 13]]
    print(matmul_2(M, N)) # [[8, 5], [20, 13]]
    print(numpy_api(M, N)) # [[8, 5], [20, 13]]
    """
    N = int(sys.argv[1])  # size of matrix
    T = int(sys.argv[2])  # num of runs
    FLOP = N * N * 2 * N  # just for square matmul
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    for _ in range(T):
        start_t = time.monotonic()
        # numpy_api(A, B)  # ~42 GFLOPS
        # matmul_1(A, B)   # ~0.004 GFLOPS
        matmul_2(A, B)     # ~0.004 GLOPS, why???
        time_elapse = time.monotonic() - start_t
        GFLOPS = FLOP / time_elapse / 1e9
        print(f'GFLOPS: {GFLOPS}')
