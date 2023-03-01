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
m = SubsetModel(full_m, [1, 5, 8, 9])
θ_gt = [-1e2, 5e1, -2e2, 1e2]
set_parameters(m, θ_gt)
sampler = GibbsSampler(m; burn_in=10 * n^2, sample_interval=n^2)
println("Sampling ground-truth...")
gs, ss = sample(sampler, 1000; progress=true)
Es = mean(ss, dims=1)[1, :]
p = full_m.motif_normalizations[1] * Es[1] / (n * (n - 1))

function sample_er(p)
    A = sprand(Bool, n, n, p)
    SparseDirectedGraph(A)
end

n_samp = 1000
ss_er = zeros(n_samp, 4)

println("Sampling Erdos-Renyi...")
@showprogress for i ∈ 1:n_samp
    set_state(m, sample_er(p))
    ss_er[i, :] = get_statistics(m)
end

Es_er = mean(ss_er, dims=1)[1, :]
println(p)
println(Es ./ Es_er)

set_parameters(m, zeros(4))
println("Fitting...")
it = 50000
lr_start = 1e-1
lr_end = 5e-3
lr = lr_start * (lr_end / lr_start) .^ ((0:it-1) ./ (it - 1))
θs, Ds = equilibrium_expectation(m, Es, 1000, lr)

# try to average away the inherent fluctuations
# in θ you get with equilibrium expectation
θ_fit = mean(θs[end-100:end, :], dims=1)[1, :]
set_parameters(m, θ_fit)
println("Sampling fitted ERGM...")
_, ss_fit = sample(sampler, 1000; progress=true)

function plot_results()
    fig = Figure()
    ax = Axis(fig[1, 1])
    ax.title = "Global Edge Parameter"
    lines!(ax, 1:size(θs, 1), θs[:, 1])
    hlines!(ax, θ_gt[1]; color=:red)
    hlines!(ax, θ_fit[1]; color=:green)
    ax = Axis(fig[2, 1])
    ax.title = "Local 021U Parameter"
    lines!(ax, 1:size(θs, 1), θs[:, 2])
    hlines!(ax, θ_gt[2]; color=:red)
    hlines!(ax, θ_fit[2]; color=:green)
    ax = Axis(fig[3, 1])
    ax.title = "Local 030T Parameter"
    lines!(ax, 1:size(θs, 1), θs[:, 3])
    hlines!(ax, θ_gt[3]; color=:red)
    hlines!(ax, θ_fit[3]; color=:green)
    ax = Axis(fig[4, 1])
    ax.title = "Local 030C Parameter"
    lines!(ax, 1:size(θs, 1), θs[:, 4])
    hlines!(ax, θ_gt[4]; color=:red)
    hlines!(ax, θ_fit[4]; color=:green)

    ax = Axis(fig[1, 2])
    ax.title = "Ground-truth (green) vs Fitted (red) Edge"
    hist!(ax, ss[:, 1], color=:green)
    hist!(ax, ss_fit[:, 1], color=:red)
    ax = Axis(fig[2, 2])
    ax.title = "Ground-truth (green) vs Fitted (red) 021U" 
    hist!(ax, ss[:, 2], color=:green)
    hist!(ax, ss_fit[:, 2], color=:red)
    ax = Axis(fig[3, 2])
    ax.title = "Ground-truth (green) vs Fitted (red) 030T" 
    hist!(ax, ss[:, 3], color=:green)
    hist!(ax, ss_fit[:, 3], color=:red)
    ax = Axis(fig[4, 2])
    ax.title = "Ground-truth (green) vs Fitted (red) 030C" 
    hist!(ax, ss[:, 4], color=:green)
    hist!(ax, ss_fit[:, 4], color=:red)
    fig
end

plot_results()
