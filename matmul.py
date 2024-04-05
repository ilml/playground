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


def matmul_3(M: np.ndarray, N : np.ndarray, BLOCK: int) -> np.ndarray:
    """tiling M"""
    ii, jj = M.shape
    _, kk = N.shape
    assert ii % BLOCK == 0
    assert jj % BLOCK == 0
    assert kk % BLOCK == 0
    C = np.zeros((ii, kk))
    # o=outer loop
    for i_o in range(0, ii, BLOCK):
        for j_o in range(0, jj, BLOCK):
            for i in range(i_o, i_o+BLOCK, 1):
                for j in range(j_o, j_o+BLOCK, 1):
                    for k in range(kk):
                        C[i, k] += M[i, j] * N[j, k]
    return C


def matmul_4(M: np.ndarray, N : np.ndarray, BLOCK: int) -> np.ndarray:
    """tiling both M and N"""
    ii, jj = M.shape
    _, kk = N.shape
    assert ii % BLOCK == 0
    assert jj % BLOCK == 0
    assert kk % BLOCK == 0
    C = np.zeros((ii, kk))
    # o=outer loop
    for i_o in range(0, ii, BLOCK):
        for j_o in range(0, jj, BLOCK):
            for k_o in range(0, kk, BLOCK):
                for i in range(i_o, i_o+BLOCK, 1):
                    for j in range(j_o, j_o+BLOCK, 1):
                        for k in range(k_o, k_o+BLOCK, 1):
                            C[i, k] += M[i, j] * N[j, k]
    return C



if __name__ == "__main__":
    N = int(sys.argv[1])  # size of matrix
    T = int(sys.argv[2])  # num of runs
    BLOCK = int(sys.argv[3])
    # tiny unit test for implementation correctness
    A = np.random.randint(-10, 10, size=(N, N))
    B = np.random.randint(-10, 10, size=(N, N))
    gt = np.matmul(A, B)
    assert np.array_equal(gt, matmul_1(A, B))
    assert np.array_equal(gt, matmul_2(A, B))
    assert np.array_equal(gt, matmul_3(A, B, BLOCK))
    assert np.array_equal(gt, matmul_4(A, B, BLOCK))


    FLOP = N * N * 2 * N  # just for square matmul
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    for _ in range(T):
        start_t = time.monotonic()
        # numpy_api(A, B)        # ~42 GFLOPS
        # matmul_1(A, B)         # ~0.004 GFLOPS
        # matmul_2(A, B)         # ~0.004 GLOPS, why???
        # matmul_3(A, B, BLOCK)  # ~0.0037 GLOPS, why??? 
        matmul_4(A, B, BLOCK)    # ~0.0035 GLOPS, why...
        time_elapse = time.monotonic() - start_t
        GFLOPS = FLOP / time_elapse / 1e9
        print(f'GFLOPS: {GFLOPS}')