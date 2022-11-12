using ergm.stats
using ergm.spaces
using ergm.inference
using ergm.sampler
using SparseArrays
using Random
using LinearAlgebra
using JLD

#@load "G_proof_v6.jld"
#n = size(node_positions, 1)
#e = size(edge_list, 1)
#A = sparse(edge_list[:, 1] .+ 1, edge_list[:, 2] .+ 1, fill(true, e), n, n)
n = 500
node_positions = rand(n, 3)



m = minimum(node_positions, dims=1)
M = maximum(node_positions, dims=1)
X = (node_positions .- m) ./ (M - m)
r = 0.2
s = LocalThreeNodeMotifStats(X, r)

#Gs, ss = sample()
#G = DiGraph(A)
#
#θs, Ls, target_Es, Es = ee(Gs, s, 10000, zeros(2), 3000, 1e4)
