using ergm.models
using ergm.sampler
using ergm.stats
using ergm.spaces
using ergm.inference
using Statistics

n = 10
θ = [-100.0]
f(G) = [
    sum(G.adjacency) / (n * (n - 1))
]
m = ExponentialFamily(SimpleStats(f), θ)
G0 = DiGraph(rand(Bool, (n, n)))
s = GibbsSampler(
    G0,
    m,
    10,
    10
)
Gs, _ = sample(s, 1000)
update_params(m, [0.0])
θs, Ls = mcmc_mle(Gs, m, 1000, 100, 10)
