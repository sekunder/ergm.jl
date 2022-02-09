using ergm.models
using ergm.sampler
using ergm.stats
using ergm.spaces
using ergm.inference
using Statistics

n = 100
θ = [-100.0]
function f(G)
    [sum(G.adjacency) / (n * (n - 1))]
end
st = DeltaStats(
    f,
    function(G, s, u)
        i, x = u
        δ_edges = x - G[i]
        ed = s[1]
        new_ed = ed + δ_edges / (n * (n - 1))
        [new_ed]
    end
)
m = ExponentialFamily(st, θ)
m2 = ExponentialFamily(SimpleStats(f), θ)
G0 = DiGraph(rand(Bool, (n, n)))
s = ParallelGibbsSampler(
    G0,
    m,
    10,
    10,
    6
)
s2 = ParallelGibbsSampler(
    G0,
    m2,
    10,
    10,
    6
)
Gs, ss = sample(s, 1000)
#update_params(m, [0.0])
#θs, Ls = mcmc_mle(Gs, m, 1000, 100, 10, 10, 10)
