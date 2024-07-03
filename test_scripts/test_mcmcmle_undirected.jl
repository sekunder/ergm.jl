using ergm
using ergm.inference
using ergm.sampling
using ergm.models
using Statistics
using SparseArrays
using LinearAlgebra
using StatsPlots

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
learning_rate = 1.0 #would like to make the learning rate start high and go down over time maybe, so we can get closer to out true params and then flatten out


println("Sampling for first round of inference:")
model = ScaffoldedEdgeTriangleModel(S, θ_gt)
sampler = GibbsSampler(model; burn_in=burn_in, sample_interval=sample_interval)
gs, ss = sample(sampler, 100)
Es = mean(ss, dims=1)[1, :]

println("Finding the Theta's using monte carlo gradient ascent(First inference):")
set_parameters(model, [0.0, 0.0])
θs_first_round = monte_carlo_gradient_ascent(
    model, Es, Dict(:burn_in => burn_in, :sample_interval => sample_interval), 100, iterations, learning_rate
)
println(θs_first_round[end, :])

println("Sampling for second round of inference from the first inferred model:")
# Sample from the inferred model
set_parameters(model, θs_first_round[end, :])
gs, ss = sample(sampler, 100)
Es_new = mean(ss, dims=1)[1, :]

println("Finding the Theta's using monte carlo gradient ascent(Second inference):")
θs_second_round = monte_carlo_gradient_ascent(
    model, Es_new, Dict(:burn_in => burn_in, :sample_interval => sample_interval), 100, iterations, learning_rate
)
println(θs_second_round[end, :])

# Plot the parameter updates for both rounds

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



rolling_window = 20 
rolling_averages_r1 = zeros(iterations - rolling_window + 1, length(θ_gt))
rolling_averages_r2 = zeros(iterations - rolling_window + 1, length(θ_gt))

for i in 1:(iterations - rolling_window + 1) #This is driving me insane
    rolling_averages_r1[i, :] = mean(θs_first_round[i:(i + rolling_window - 1), :], dims=1)
    rolling_averages_r2[i, :] = mean(θs_second_round[i:(i + rolling_window - 1), :], dims=1)
end

p = plot(1:iterations, θs_first_round[:, 1], label="Parameter 1 (Unit)", xlabel="Iteration", ylabel="Parameter Value", title="Parameter Convergence with Unit Step")
plot!(p, 1:iterations, θs_first_round[:, 2], label="Parameter 2 (Unit)")
plot!(p, 1:iterations, θs_second_round[:, 1], label="Parameter 1 (2nd round)")
plot!(p, 1:iterations, θs_second_round[:, 2], label="Parameter 2 (2nd round)")

plot!(p, rolling_window:iterations, rolling_averages_r1[:, 1], label="Rolling Avg Parameter 1", linestyle=:dash)
plot!(p, rolling_window:iterations, rolling_averages_r1[:, 2], label="Rolling Avg Parameter 2", linestyle=:dash)
plot!(p, rolling_window:iterations, rolling_averages_r2[:, 1], label="Rolling Avg Parameter 1", linestyle=:dash)
plot!(p, rolling_window:iterations, rolling_averages_r2[:, 2], label="Rolling Avg Parameter 2", linestyle=:dash)

hline!(p, [θ_gt[1]], label="True Parameter 1", linestyle=:dot, color=:red)
hline!(p, [θ_gt[2]], label="True Parameter 2", linestyle=:dot, color=:blue)

display(p)

println("Plotting: ")
p = plot(1:iterations, θs_first_round[:, 1], label="Estimated Parameter 1 (1st round)", xlabel="Iteration", ylabel="Parameter Value", title="Parameter Convergence with Unit Step (Multiple Rounds)")
plot!(p, 1:iterations, θs_first_round[:, 2], label="Estimated Parameter 2 (1st round)")

plot!(p, 1:iterations, θs_second_round[:, 1], label="Estimated Parameter 1 (2nd round)", linestyle=:dash)
plot!(p, 1:iterations, θs_second_round[:, 2], label="Estimated Parameter 2 (2nd round)", linestyle=:dash)

# Plot true parameters as horizontal lines
hline!(p, [θ_gt[1]], label="True Parameter 1", linestyle=:dot, color=:red)
hline!(p, [θ_gt[2]], label="True Parameter 2", linestyle=:dot, color=:blue)

display(p)


