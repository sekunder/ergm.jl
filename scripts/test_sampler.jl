using ergm.spaces
using ergm.models
using ergm.stats
using ergm.sampler
using Statistics
using Random

# sampling from ground-truth with over-represented local triangles
n = 50
θ = [0e0, 1e4]
Ω = SparseGraphs(n)
X = rand(n, 3)
r = 0.5
model = ERGM(SparseGraphs(50), LocalEdgeTriangle, θ, X, r)
sampler = GibbsSampler(model, burn_in=10*n^2, sample_interval=n^2)
Gs, ss = sample(sampler, 1000)
Es = mean(ss, dims=2)
println("$θ -> $Es")

# sampling from Erdos-Renyi ground-truth with similar edge density
n = 50
θ = [5e2, 0e0]
Ω = SparseGraphs(n)
X = rand(n, 3)
r = 0.5
model = ERGM(SparseGraphs(50), LocalEdgeTriangle, θ, X, r)
sampler = GibbsSampler(model, burn_in=10*n^2, sample_interval=n^2)
Gs, ss = sample(sampler, 1000)
Es = mean(ss, dims=2)
println("$θ -> $Es")
