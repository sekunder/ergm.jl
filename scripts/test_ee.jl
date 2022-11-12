using ergm.stats
using ergm.models
using ergm.sampler
using ergm.spaces
using ergm.optim
using ergm.inference
using Statistics
using LinearAlgebra

n = 500
node_locations = rand(3, n)
node_metric = [
    norm(node_locations[i] - node_locations[j])
    for i ∈ 1:n, j ∈ 1:n
]
local_radius = 0.2
local_mask = node_metric .< local_radius

max_edge_count = n * (n - 1) / 2
max_local_triangle_count = sum(local_mask ^ 2 .* local_mask) / 6

sufficient_statistics = DeltaStats(
    function(G)
        n = G.n
        A = G.adjacency
        
        edge_count = sum(A) / 2
        edge_density = edge_count / max_edge_count
        
        local_A = local_mask .* A
        local_triangle_count = sum(local_A ^ 2 .* local_A) / 6
        local_triangle_density = local_triangle_count / max_local_triangle_count
        
        [edge_density, local_triangle_density]
    end,
    function(G, current_statistics, update)
        n = G.n
        A = G.adjacency
        ix, x = update
        i, j = ix
        
        delta_edge_count = x - G[ix]
        delta_edge_density = delta_edge_count / max_edge_count
        
        local_A = local_mask .* A
        delta_local_triangle_count = (x - G[ix]) * local_mask[i, j] * sum(local_A[i, :] .* local_A[:, j])
        delta_local_triangle_density = delta_local_triangle_count / max_local_triangle_count
        
        current_statistics .+ [delta_edge_density, delta_local_triangle_density]
    end
)

θ_gt = [-1e3, 1e3]
model = ExponentialFamily(sufficient_statistics, θ_gt)
sampler = ParallelGibbsSampler(
    Graph(zeros(Bool, n, n)), model, 3, 5, Threads.nthreads()
)
Gs, _ = sample(sampler, 100)
θs, Ls, target_Es, Es = inference.ee(Gs, model, 1000, zeros(2), 100, 1e4)