# some sample algo here

"""
importance sample:
the expectation of f given p is Ep[f], if p is hard to sample, we can sample under q, which is Eq[p*f/q]

rejection sample:
we want to sample from p(x) = f(x) / NC, where f(x) is easy to sample
we propose another distribution g(x) and M, where f(x) < M*g(x) 

1. sample from g(x)
2. accept with prob [f(x)/M*g(x)]

MCMC:
1. sample based on previous sample
2. construct a markov chain that satifies detailed balance(pT=p)


M-H:
1. sample from g(xt+1|xt) , e.g. g = N(xt, sigma)
2. accept xt+1 with prob: min(1, rfrg), where rf = f(b) / f(a), fg = g(a|b) / g(b|a)


"""





