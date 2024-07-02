using ergm
using ergm.inference
using ergm.sampling
using ergm.models
using Statistics
using SparseArrays
using LinearAlgebra

half_n = 20
n = 2 * half_n

S = [ones(half_n, half_n) zeros(half_n, half_n);
     zeros(half_n, half_n) ones(half_n, half_n)]
for i in 1:n
    S[i, i] = 0
end

nodes = n
θ_gt = [-1.0, 50.0] 
burn_in = 10 * nodes^2
sample_interval = nodes^2
iterations = 100
step_size = 50

model = ScaffoldedEdgeTriangleModel(S, θ_gt)

sampler = GibbsSampler(model; burn_in=burn_in, sample_interval=sample_interval)

println("Sampling:")

gs, ss = sample(sampler, 100)

Es = mean(ss, dims=1)[1, :]

set_parameters(model, [0.0, 0.0])

println("Finding Theta's using monte carlo gradient ascent:")

θs = monte_carlo_gradient_ascent(
    model, Es, Dict(:burn_in => burn_in, :sample_interval => sample_interval), iterations, step_size, 1.0
)

println("Recovered parameters: ", θs[end,:])

# set_parameters(model, [0.0, 0.0])

# println("Finding Theta's using monte carlo gradient ascent - Hessian")

# θs_hessian = ergm.inference.monte_carlo_gradient_ascent_hessian(
#     model, Es, Dict(:burn_in => burn_in, :sample_interval => sample_interval), iterations, step_size, 0.5
# )

# println("Recovered parameters (Hessian):" , θs_hessian[end, :])

# plot(
#     1:length(θ_gt), θ_gt, label="Original Parameters", lw=2, marker=:circle,     xlabel="Parameter Index", ylabel="Parameter Value", title="Original vs Sampled Parameters"
# )
# plot!(1:length(θs), θs, label="Sampled Parameters", lw=2, marker=:star)

# display(plot)




