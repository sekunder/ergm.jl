using ergm.inference
using ergm.sampling
using ergm.models
using ergm.spaces
using Statistics
using SparseArrays
using GLMakie
using ProgressMeter

# sample from ground-truth edge-density ERGM and
# try to recover natural parameter via EE
n = 200
X = randn((n, 3))
r = 1.0
full_m = DirectedSpatialTripletModel(X, r, zeros(15))
m = SubsetModel(full_m, collect(1:14))
θ_gt = [-3e2, 5e1, -5e0, -2e2, 8e1, 5e1, -5e1, 1e2, -1e2, 5e1, -5e1, 1e2, 1e2, -1e2]
set_parameters(m, θ_gt)
sampler = GibbsSampler(m; burn_in=10 * n^2, sample_interval=n^2)
println("Sampling ground-truth...")
gs, ss = sample(sampler, 1000; progress=true)
Es = mean(ss, dims=1)[1, :]
p = full_m.motif_normalizations[1] * Es[1] / (n * (n - 1))
println("Edge density: $p")

function sample_er(p)
    A = sprand(Bool, n, n, p)
    SparseDirectedGraph(A)
end

n_samp = 1000
ss_er = zeros(n_samp, 14)

println("Sampling Erdos-Renyi...")
@showprogress for i ∈ 1:n_samp
    set_state(m, sample_er(p))
    ss_er[i, :] = get_statistics(m)
end

Es_er = mean(ss_er, dims=1)[1, :]
println(Es ./ Es_er)

set_parameters(m, zeros(14))
println("Fitting...")
it = 40000
lr_start = 5e-2
lr_end = 2e-3
lr = lr_start * (lr_end / lr_start) .^ ((0:it-1) ./ (it - 1))
θs, Ds, fs = equilibrium_expectation(m, Es, 5000, lr)

# try to average away the inherent fluctuations
# in θ you get with equilibrium expectation
θ_fit = mean(θs[end-100:end, :], dims=1)[1, :]
set_parameters(m, θ_fit)
println("Sampling fitted ERGM...")
_, ss_fit = sample(sampler, 1000; progress=true)
Es_fit = mean(ss_fit; dims=1)[1, :]

function plot_results(ps)
    param_names = ["012", "102", "021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D", "120U", "120C", "210"]
    fig = Figure()

    i = 1

    for p ∈ ps
        ax = Axis(fig[i, 1])
        ax.title = "Natural Parameters θ[$p] = $(param_names[p])"
        θs_p = θs[10000:end, p]
        lines!(ax, 1:length(θs_p), θs_p)
        hlines!(ax, θ_gt[p]; color=:red)
        hlines!(ax, θ_fit[p]; color=:green)

        ax = Axis(fig[i, 2])
        ax.title = "Ground-truth (green) vs Fitted (red) $(param_names[p])"
        hist!(ax, ss[:, p], color=:green)
        hist!(ax, ss_fit[:, p], color=:red)
        
        i += 1
    end
    fig
end

plot_results([1,2])
plot_results([3,4])
plot_results([5,6])
plot_results([7,8])
plot_results([9,10])
plot_results([11,12])
plot_results([13,14])
