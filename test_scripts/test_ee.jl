using ergm.inference
using ergm.sampling
using ergm.models
using Statistics

nodes = 30
edge_density(x) = sum(x.adjacency) / (nodes * (nodes - 1))
θ_gt = [17.0]
model = SimpleModel(nodes, [edge_density], θ_gt)
sampler = GibbsSampler(model; burn_in=10 * nodes^2, sample_interval=nodes^2)
gs, ss = sample(sampler, 100)
Es = mean(ss, dims=1)[1, :]

set_parameters(model, [0.0])
θs = monte_carlo_gradient_ascent(
    model, Es, Dict(:burn_in => 10 * nodes^2, :sample_interval => nodes^2), 100, 50, 1e2
)
