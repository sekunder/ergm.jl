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

scaffold = data.example_graph("larvalMB0.0")
larvalMB = data.example_graph("larvalMB")
n = 365

model = ScaffoldedEdgeTriangleModel(scaffold, [0.0, 0.0])

target_stats = [sum(larvalMB)/2, tr(scaffold^3)/n]

burn_in = 10 * n^2
sample_interval = n^2
iterations = 100
learning_rate = 1.0 


thetas = inference.monte_carlo_gradient_ascent(
    model, target_stats,
    Dict(:burn_in => burn_in, :sample_interval => sample_interval), 
    100, iterations, learning_rate
)