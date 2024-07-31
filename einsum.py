# practice einsum on attention
import torch
import math
import torch.nn.functional as F
import time


def multi_head_attention(X, W_Q, W_K, W_V, W_O, num_head):
    B, S, H = X.shape
    H_d = H // num_head
    Q = torch.matmul(X, W_Q).view(B, num_head, S, H_d)
    K = torch.matmul(X, W_K).view(B, num_head, S, H_d)
    V = torch.matmul(X, W_V).view(B, num_head, S, H_d)
    score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(H_d)
    score = F.softmax(score.float(), dim=-1).type_as(Q)
    output = torch.matmul(score, V).view(B, S, H)
    output = torch.matmul(output, W_O)
    return output 


def multi_head_attention_einsum(X, W_Q, W_K, W_V, W_O, num_head):
    B, S, H = X.shape
    H_d = H // num_head
    Q = torch.einsum('bsh,hk->bsk', X, W_Q).view(B, num_head, S, H_d)
    K = torch.einsum('bsh,hk->bsk', X, W_K).view(B, num_head, S, H_d)
    V = torch.einsum('bsh,hk->bsk', X, W_V).view(B, num_head, S, H_d)
    score = torch.einsum('bnsh,bnhk->bnsk', Q, K.transpose(2, 3)) / math.sqrt(H_d)
    score = F.softmax(score.float(), dim=-1).type_as(Q)
    output = torch.einsum('bnsk,bnkh->bnsh', score, V).view(B, S, H)
    output = torch.einsum('bsh,hk->bsk',output, W_O)
    return output 


def benchmark(func, *args, n_runs=100):
    times = []
    for i in range(n_runs):
        print(i)
        start = time.time()
        _ = func(*args)
        torch.cuda.synchronize()  # Ensure GPU computation is done
        end = time.time()
        times.append(end - start)
    return sum(times) / len(times)

def warmup(func, *args):
    for _ in range(10):  # Run a few times to warm up
        _ = func(*args)
    torch.cuda.synchronize()  # Ensure all GPU computations are finished


if __name__ == '__main__':
    B, S, H, num_head = 640, 640, 640, 8
    X = torch.rand(B, S, H).cuda()
    W_Q = torch.rand(H, H).cuda()
    W_K = torch.rand(H, H).cuda()
    W_V = torch.rand(H, H).cuda()
    W_O = torch.rand(H, H).cuda()
    r1 = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_head)
    r2 = multi_head_attention_einsum(X, W_Q, W_K, W_V, W_O, num_head)
    assert torch.allclose(r1, r2)

  
    # Warm up both functions
    warmup(multi_head_attention, X, W_Q, W_K, W_V, W_O, num_head)
    warmup(multi_head_attention_einsum, X, W_Q, W_K, W_V, W_O, num_head)
    
    # Benchmark both functions
    time1 = benchmark(multi_head_attention, X, W_Q, W_K, W_V, W_O, num_head)
    time2 = benchmark(multi_head_attention_einsum, X, W_Q, W_K, W_V, W_O, num_head)

    print(f"Original version: {time1:.6f} seconds per run")
    print(f"Einsum version: {time2:.6f} seconds per run")