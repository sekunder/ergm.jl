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

scaffold = data.example_graph("larvalMB0.5")
larvalMB = data.example_graph("larvalMB")
n = 365

small_scaffold = scaffold[1:50, 1:50]
small_larvalMB = larvalMB[1:50, 1:50]
small_n = 50

model = ScaffoldedEdgeTriangleModel(small_scaffold, [0.0, 0.0])

target_stats = [sum(small_larvalMB)/2, tr(small_scaffold^3)/small_n]
real_density = [sum(small_larvalMB)/(small_n*(small_n-1)), tr((small_larvalMB)^3)/((small_n) * (small_n-1) *(small_n-2))]
scaffold_density = [sum(small_scaffold)/(small_n*(small_n-1)), tr((small_scaffold)^3)/((small_n) * (small_n-1) *(small_n-2))]

burn_in = 10 * small_n^2
sample_interval = small_n^2
iterations = 50  
learning_rate = 1.0 
verbose = true  

log_message("Starting Monte Carlo Gradient Ascent...", verbose)

thetas = inference.monte_carlo_gradient_ascent(
    model, target_stats,
    Dict(:burn_in => burn_in, :sample_interval => sample_interval), 
    100, iterations, learning_rate
)

log_message("Monte Carlo Gradient Ascent completed.", verbose)
println("Final thetas: ", thetas[end, :])

# set_parameters(model, thetas[end, :])
new_model = ScaffoldedEdgeTriangleModel(small_scaffold, thetas[end, :])

new_sampler = GibbsSampler(new_model; burn_in=burn_in, sample_interval=sample_interval)
new_graphs, new_stats = ergm.sampling.sample(new_sampler, 100, progress = true)

As = [matrix_form(G) for G in new_graphs]
Ss = [G.scaffold_edges for G in new_graphs]

densities = cat([[sum(A) / (n * (n-1)) tr(A^3) / (n * (n-1) * (n-2))] for A in As]..., dims=1)
scaff_only_densities = cat([[sum(M) / (n * (n-1)) tr(M^3) / (n * (n-1) * (n-2))] for M in Ss]..., dims=1)

xs = 0.01:0.01:1
er_curve = xs .^ 3
er_upper = sqrt.(xs) .^ 3
xs_lower = 0.5:0.0001:1
er_lower = xs_lower .* (2xs_lower .- 1)

p = plot(xs, er_curve, color="gray", label="E-R", legend=:topleft, title="Densities with Recovered Parameters")
plot!(p, xs, er_upper, color="gray", linestyle=:dash, label=false)
plot!(p, xs_lower[2:end], er_lower[2:end], color="gray", linestyle=:dash, label=false)
scatter!(p, densities[:,1], densities[:,2], label="Samples")
scatter!(p, scaff_only_densities[:,1], scaff_only_densities[:,2], marker=:x, label="Scaffold")
scatter!(p, [real_density[1]], [real_density[2]], markershape = :star, markersize = 5, label="Original Graph Densities")
scatter!(p, [scaffold_density[1]], [scaffold_density[2]], markershape = :star, markersize = 5, label="Original Scaffold Densities")

display(p)
