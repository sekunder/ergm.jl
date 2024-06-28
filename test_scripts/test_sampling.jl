using ergm
using ergm.models
using ergm.sampling

using Pkg
Pkg.add("StatsPlots")
Pkg.add("Distributions")
Pkg.add("StatsBase")

using StatsPlots
using Distributions
using StatsBase

half_n = 25
n = 2 * half_n

S = [ones(half_n, half_n) zeros(half_n, half_n);
     zeros(half_n, half_n) ones(half_n, half_n)]
for i in 1:n
    S[i,i] = 0
end

M = ScaffoldedEdgeTriangleModel(S, [0.0, -100.0])
sampler = GibbsSampler(M, burn_in=n ^ 3, sample_interval=10 * n^2)
graphs, stats = sample(sampler, 100, progress=true)

scaled_stats = stats .* [1 n]

autocor_values = autocor(scaled_stats[:, 1])

# Plot the autocorrelation function
plot(autocor_values, seriestype=:stem, title="Autocorrelation of Gibbs Sampler Statistics", xlabel="Lag", ylabel="Autocorrelation")

# Function to calculate rolling mean
function rolling_mean(data, window_size)
    n = length(data)
    rolling_means = [mean(data[max(1, i-window_size+1):i]) for i in 1:n]
    return rolling_means
end

# Compute rolling means for the first statistic (or any other of interest)
window_size = 10  # Set window size for rolling mean
rolling_means = rolling_mean(scaled_stats[:, 1], window_size)

# Plot the rolling means
plot(rolling_means, title="Rolling Mean of Gibbs Sampler Statistics", xlabel="Sample Index", ylabel="Rolling Mean", label="Window Size = $window_size")