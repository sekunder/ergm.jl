using ergm.stats
using ergm.spaces
using ergm.inference
using ergm.sampler
using SparseArrays
using Random
using LinearAlgebra
using JLD

@load "G_proof_v6.jld"
n = size(node_positions, 1)
e = size(edge_list, 1)
m = minimum(node_positions, dims=2)
M = maximum(node_positions, dims=2)
X = (node_positions .- m) ./ (M - m)
r = 0.2
s = LocalThreeNodeMotifStats(X, r)
gs = GibbsSampler(s, [0.0, 0.0], 10 * n, n)
ss = sample_stats(gs, 1)
