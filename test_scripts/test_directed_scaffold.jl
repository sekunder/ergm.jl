using ergm
using ergm.models
using ergm.sampling
using ergm.spaces

using StatsPlots
using Distributions
using StatsBase
using LinearAlgebra

block_size = 20
n_blocks = 2
n = n_blocks * block_size
total_scaffold_edges = n_blocks * block_size * (block_size-1) ÷ 2

S = zeros(n,n)
for block in 1:n_blocks
    indices = 1+(block-1)*block_size:block*block_size
    S[indices, indices] .= 1
end
for i in 1:n
    S[i,i] = 0
end


# θtrue = zeros(15)  # should be an erdos-renyi graph with p = 0.5, I think
θtrue = fill(1.0, 15)
θtrue[1] = -10.0
θtrue[2] = -2.0
θtrue[8] = 2.0
ERGM = ScaffoldedTripletModel(S, θtrue, false)
g0 = copy(ERGM.state)

n_samples = 100
sampler = GibbsSampler(ERGM, burn_in=n^3, sample_interval = 10 * n^2)
graphs, stats = ergm.sampling.sample(sampler, n_samples, progress=true)

motif_counts = (stats .* n)

# let's check if we miscounted the motifs at any point
for graph_index in 1:100
    g1 = graphs[graph_index]
    s1 = motif_counts[graph_index,:]

    A1 = adjacency_matrix(g1)
    computed_stats = triplet_motif_counts(S .* A1)
    computed_stats[1] = sum(A1)
    computed_stats[2] = sum(A1 .* A1') ÷ 2

    if any(computed_stats .!= s1)
        print("Graph $graph_index has a mismatch")
    end
end

begin
motif_densities = motif_counts ./ [n*(n-1) n*(n-1)/2 fill(n*(n-1)*(n-2)/6, (1,13))]
rhos = motif_densities[:, 1]
expected_densities = zeros(size(motif_densities))
# expected_densities = [rhos rhos.^2 rhos.^2 .* (1 .- rhos).^4]
expected_densities[:, 1] = rhos
expected_densities[:, 2] = rhos.^2
expected_densities[:, 3] = rhos.^2 .* (1 .- rhos).^4
expected_densities[:, 4] = rhos.^2 .* (1 .- rhos).^4
expected_densities[:, 5] = rhos.^2 .* (1 .- rhos).^4
expected_densities[:, 6] = rhos.^3 .* (1 .- rhos).^3
expected_densities[:, 7] = rhos.^3 .* (1 .- rhos).^3
expected_densities[:, 8] = rhos.^3 .* (1 .- rhos).^3
expected_densities[:, 9] = rhos.^3 .* (1 .- rhos).^3
expected_densities[:,10] = rhos.^4 .* (1 .- rhos).^2
expected_densities[:,11] = rhos.^4 .* (1 .- rhos).^2
expected_densities[:,12] = rhos.^4 .* (1 .- rhos).^2
expected_densities[:,13] = rhos.^4 .* (1 .- rhos).^2
expected_densities[:,14] = rhos.^5 .* (1 .- rhos).^1
expected_densities[:,15] = rhos.^6
end

bar(mean(motif_densities, dims=1)[:], label="Sample Mean",
    yaxis=:log)
plot!(mean(expected_densities, dims=1)[:], label="Expected",
     yaxis=:log)