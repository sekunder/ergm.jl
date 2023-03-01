using ergm.inference
using ergm.sampling
using ergm.models
using ergm.spaces
using Statistics
using SparseArrays
using GLMakie

# sample from ground-truth edge-density ERGM and
# try to recover natural parameter via EE
n = 100
X = randn((n, 3))
r = 1.0
full_m = DirectedSpatialTripletModel(X, r, zeros(15))
m = SubsetModel(full_m, [1, 7])
θ_gt = [-2e2, 4e1]
set_parameters(m, θ_gt)
sampler = GibbsSampler(m; burn_in=10 * n^2, sample_interval=n^2)
gs, ss = sample(sampler, 1000; progress=true)
Es = mean(ss, dims=1)[1, :]
p = full_m.motif_normalizations[1] * Es[1] / (n * (n - 1))

function sample_er(p)
    A = sprand(Bool, n, n, p)
    SparseDirectedGraph(A)
end

n_samp = 100
ss_er = zeros(n_samp, 2)

for i ∈ 1:n_samp
    set_state(m, sample_er(p))
    ss_er[i, :] = get_statistics(m)
end

Es_er = mean(ss_er, dims=1)[1, :]
println(p)
println(Es ./ Es_er)

set_parameters(m, zeros(2))
θs, Ds = equilibrium_expectation(m, Es, 1000, 10000, 1e1)

# try to average away the inherent fluctuations
# in θ you get with equilibrium expectation
θ_fit = mean(θs[end-100:end, :], dims=1)[1, :]
set_parameters(m, θ_fit)
_, ss_fit = sample(sampler, 1000; progress=true)

function plot_results()
    fig = Figure()
    ax1 = Axis(fig[1, 1])
    ax1.title = "Global Edge Parameter"
    lines!(ax1, 1:size(θs, 1), θs[:, 1])
    hlines!(ax1, θ_gt[1]; color=:red)
    hlines!(ax1, θ_fit[1]; color=:green)
    ax2 = Axis(fig[2, 1])
    ax2.title = "Local 3-Cycle Parameter"
    lines!(ax2, 1:size(θs, 1), θs[:, 2])
    hlines!(ax2, θ_gt[2]; color=:red)
    hlines!(ax2, θ_fit[2]; color=:green)
    ax3 = Axis(fig[1, 2])
    ax3.title = "Ground-truth (green) vs Fitted (red) Edge"
    hist!(ax3, ss[:, 1], color=:green)
    hist!(ax3, ss_fit[:, 1], color=:red)
    ax3 = Axis(fig[2, 2])
    ax3.title = "Ground-truth (green) vs Fitted (red) 3-Cycle"
    hist!(ax3, ss[:, 2], color=:green)
    hist!(ax3, ss_fit[:, 2], color=:red)
    fig
end

plot_results()
