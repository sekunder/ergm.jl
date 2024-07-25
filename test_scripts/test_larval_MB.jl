using Pkg
Pkg.add("StatsPlots")
Pkg.add("LaTeXStrings")
Pkg.add(url="https://github.com/sekunder/ergm.jl")

using ergm
using ergm.inference
using ergm.sampling
using ergm.models
using ergm.spaces
using ergm.data
using Statistics
using SparseArrays
using LinearAlgebra
using StatsPlots
using LaTeXStrings

function matrix_form(G::ScaffoldedUndirectedGraph)
    A = copy(G.scaffold_edges)
    for edge in G.other_edges
        A[edge...] = true
        A[reverse(edge)...] = true
    end
    return A
    
end

function log_message(message, verbose)
    if verbose
        println(message)
    end
end

verbose = true
progress_bars = false
larvalMB = data.example_graph("larvalMB")
n = 365
log_message("Using $n / 365 nodes", verbose)

# chi = "0.25"
for chi in ["0.0", "0.05", "0.1", "0.25", "0.5", "0.75", "1.0"]
    log_message("Processing chi = $chi", verbose)
    scaffold = data.example_graph("larvalMB" * chi)

    small_scaffold = scaffold[1:n, 1:n]
    small_larvalMB = larvalMB[1:n, 1:n]

    model = ScaffoldedEdgeTriangleModel(small_scaffold, [0.0, 0.0])

    target_stats = [sum(small_larvalMB)/2, tr(small_scaffold^3)/n]
    real_density = [sum(small_larvalMB)/(n*(n-1)), tr((small_larvalMB)^3)/((n) * (n-1) *(n-2))]
    scaffold_density = [sum(small_scaffold)/(n*(n-1)), tr((small_scaffold)^3)/((n) * (n-1) *(n-2))]

    burn_in = 10 * n^2
    sample_interval = n^2
    iterations = 50  
    learning_rate = 1.0 
    

    log_message("Starting Monte Carlo Gradient Ascent...", verbose)

    thetas = inference.monte_carlo_gradient_ascent(
        model, target_stats,
        Dict(:burn_in => burn_in, :sample_interval => sample_interval), 
        100, iterations, learning_rate,
        progress=progress_bars
    )

    log_message("Monte Carlo Gradient Ascent completed.", verbose)
    println("Thetas:")
    println()
    println(thetas)
    println()
    println("Final thetas: ", thetas[end, :])

    # set_parameters(model, thetas[end, :])
    new_model = ScaffoldedEdgeTriangleModel(small_scaffold, thetas[end, :])

    log_message("Sampling using inferred parameters", verbose)
    new_sampler = GibbsSampler(new_model; burn_in=burn_in, sample_interval=sample_interval)
    new_graphs, new_stats = ergm.sampling.sample(new_sampler, 100, progress=progress_bars)

    As = [matrix_form(G) for G in new_graphs]
    Ss = [G.scaffold_edges for G in new_graphs]

    densities = cat([[sum(A) / (n * (n-1))  tr(A^3) / (n * (n-1) * (n-2))] for A in As]..., dims=1)
    scaff_only_densities = cat([[sum(M) / (n * (n-1))  tr(M^3) / (n * (n-1) * (n-2))] for M in Ss]..., dims=1)

    xs = 0.01:0.01:1
    er_curve = xs .^ 3
    er_upper = sqrt.(xs) .^ 3
    xs_lower = 0.5:0.0001:1
    er_lower = xs_lower .* (2xs_lower .- 1)

    p = plot(xs, er_curve, color="gray", label="E-R", legend=:topleft,
            title="Densities with Recovered Parameters, chi=$chi\n$(thetas[end,:])")
    plot!(p, xs, er_upper, color="gray", linestyle=:dash, label=false)
    plot!(p, xs_lower[2:end], er_lower[2:end], color="gray", linestyle=:dash, label=false)
    scatter!(p, densities[:,1], densities[:,2], label="Samples")
    scatter!(p, scaff_only_densities[:,1], scaff_only_densities[:,2], marker=:x, label="Scaffold")
    scatter!(p, [real_density[1]], [real_density[2]], markershape = :star, markersize = 5, label="Original Graph Densities")
    scatter!(p, [scaffold_density[1]], [scaffold_density[2]], markershape = :star, markersize = 5, label="Original Scaffold Densities")
    xlabel!("Edge Density")
    ylabel!("Triangle Density")

    savefig(p, "larval_MB_"*chi*"_densities.pdf")

    trajectory_fig = plot(thetas[:,1], thetas[:,2],
                        title="Theta trajectory", label=L"\theta_t",
                        xlabel=L"\theta_1", ylabel=L"\theta_2")
    # scatter!(thetas[:,1], thetas[:,2], zcolor=1:iterations, label=false)

    savefig(trajectory_fig, "larval_MB_"*chi*"_trajectory.pdf")

    log_message("Figures saved to " * pwd(), verbose)

    # save p, trajectory_fig

    # display(p)
end