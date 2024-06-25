using ergm
using ergm.models
using ergm.sampling
using ergm.spaces

# using Pkg
# Pkg.add("StatsPlots")
# Pkg.add("Distributions")
# Pkg.add("StatsBase")

using StatsPlots
using Distributions
using StatsBase
using LinearAlgebra

function matrix_form(G::ScaffoldedUndirectedGraph)
    A = copy(G.scaffold_edges)
    for edge in G.other_edges
        A[edge...] = true
        A[reverse(edge)...] = true
    end
    return A
end

half_n = 20
n = 2 * half_n

S = [ones(half_n, half_n) zeros(half_n, half_n);
     zeros(half_n, half_n) ones(half_n, half_n)]
for i in 1:n
    S[i,i] = 0
end

M = ScaffoldedEdgeTriangleModel(S, [1.0, -50.0])
M.state
sampler = GibbsSampler(M, burn_in=n^3, sample_interval=10 * n^2)
graphs, stats = ergm.sampling.sample(sampler, 100, progress=true)

scaled_stats = stats .* [1 n]

# autocor_values_e = autocor(scaled_stats[:, 1], 0:100)
# autocor_values_t = autocor(scaled_stats[:, 2], 0:100)

# # Plot the autocorrelation function
# plot(autocor_values_e, title="Autocorrelation of Gibbs Sampler Statistics", xlabel="Lag", ylabel="Autocorrelation")
# plot(autocor_values_t, title="Autocorrelation of Gibbs Sampler Statistics", xlabel="Lag", ylabel="Autocorrelation")

# # Function to calculate rolling mean
# function rolling_mean(data, window_size)
#     n = length(data)
#     rolling_means = [mean(data[max(1, i-window_size+1):i]) for i in 1:n]
#     return rolling_means
# end

# # Compute rolling means for the first statistic (or any other of interest)
# window_size = 100  # Set window size for rolling mean
# rolling_means_e = rolling_mean(scaled_stats[:, 1], window_size)
# rolling_means_t= rolling_mean(scaled_stats[:, 2], window_size)

# # Plot the rolling means
# plot(rolling_means_e, title="Rolling Mean of Gibbs Sampler Statistics", xlabel="Sample Index", ylabel="Rolling Mean", label="Window Size = $window_size")
# plot(rolling_means_t, title="Rolling Mean of Gibbs Sampler Statistics", xlabel="Sample Index", ylabel="Rolling Mean", label="Window Size = $window_size")




As = [matrix_form(G) for G in graphs]
densities = cat([[sum(A) / n^2 tr(A^3) / n^3] for A in As]..., dims=1)

xs = 0:0.01:1
er_curve = xs .^ 3
er_upper = sqrt.(xs) .^ 3
er_lower = max.(xs .* (2xs .- 1), 0)
plot(xs, er_curve, color="gray")
plot!(xs, er_upper, color="gray", linestyle=:dash)
plot!(xs, er_lower, color="gray", linestyle=:dash)
scatter!(densities[:,1], densities[:,2])
scaffolded_densities = scaled_stats ./ [(n^2 / 2) (n^3 /6)]
scatter!(scaffolded_densities[:,1], scaffolded_densities[:,2], marker=:x)