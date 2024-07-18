using ergm
using ergm.inference
using ergm.sampling
using ergm.models
using ergm.spaces
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

half_n = 20
n = 2 * half_n

S = [ones(half_n, half_n) zeros(half_n, half_n);
     zeros(half_n, half_n) ones(half_n, half_n)]
for i in 1:n
    S[i, i] = 0
end

# print(S)

nodes = n
θ_gt = [-1.0, 50.0] 
burn_in = 10 * nodes^2
sample_interval = nodes^2
iterations = 100
learning_rate = 1.0 


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

p7 = plot(θs_first_round[:, 1], θs_first_round[:, 2])
scatter!(p7, θs_first_round[:, 1], θs_first_round[:, 2], zcolor=1:iterations)

println("Sampling for second round of inference from the first inferred model:") 
set_parameters(model, θs_first_round[end, :])
gs, ss = sample(sampler, 100)
Es_new = mean(ss, dims=1)[1, :]

println("Finding the Theta's using monte carlo gradient ascent(Second inference):")
θs_second_round = monte_carlo_gradient_ascent(
    model, Es_new, Dict(:burn_in => burn_in, :sample_interval => sample_interval), 100, iterations, learning_rate
)
println(θs_second_round[end, :])



# Actual stats and how they compare to E-R

p1 = plot(1:iterations, θs_first_round[:, 1], label="Estimated Parameter 1", xlabel="Iteration", ylabel="Parameter Value", title="Parameter Convergence with Unit Step")
plot!(p1, 1:iterations, θs_first_round[:, 2], label="Estimated Parameter 2")

# Plot true parameters as horizontal lines
hline!(p1, [θ_gt[1]], label="True Parameter 1", linestyle=:dot, color=:red)
hline!(p1, [θ_gt[2]], label="True Parameter 2", linestyle=:dot, color=:blue)

display(p1)

# Convert already sampled graphs to adjacency matrices
As = [matrix_form(G) for G in gs]
Ss = [G.scaffold_edges for G in gs]

# Compute densities for graphs from θs_first_round
densities = cat([[sum(A) / (n * (n-1)) tr(A^3) / (n * (n-1) * (n-2))] for A in As]..., dims=1)
scaff_only_densities = cat([[sum(M) / (n * (n-1)) tr(M^3) / (n * (n-1) * (n-2))] for M in Ss]..., dims=1)

# Sample new graphs using θ_gt = [-1.0, 50.0]
set_parameters(model, θ_gt)
new_sampler_gt = GibbsSampler(model; burn_in=burn_in, sample_interval=sample_interval)
new_graphs_gt, _ = sample(new_sampler_gt, 100)

# Convert graphs to adjacency matrices
As_gt = [matrix_form(G) for G in new_graphs_gt]
Ss_gt = [G.scaffold_edges for G in new_graphs_gt]

# Compute densities for graphs from θ_gt
densities_gt = cat([[sum(A) / (n * (n-1)) tr(A^3) / (n * (n-1) * (n-2))] for A in As_gt]..., dims=1)
scaff_only_densities_gt = cat([[sum(M) / (n * (n-1)) tr(M^3) / (n * (n-1) * (n-2))] for M in Ss_gt]..., dims=1)

# Plot combined densities
xs = 0.01:0.01:1
er_curve = xs .^ 3
er_upper = sqrt.(xs) .^ 3
xs_lower = 0.5:0.0001:1
er_lower = xs_lower .* (2xs_lower .- 1)

p2 = plot(xs, er_curve, color="gray", label="E-R", legend=:topleft, title="Combined Densities")
plot!(p2, xs, er_upper, color="gray", linestyle=:dash, label=false)
plot!(p2, xs_lower[2:end], er_lower[2:end], color="gray", linestyle=:dash, label=false)
scatter!(p2, densities[:,1], densities[:,2], label="Samples (θs_first_round)", color=:red)
scatter!(p2, scaff_only_densities[:,1], scaff_only_densities[:,2], marker=:x, label="Scaffold (θs_first_round)", color=:red)
scatter!(p2, densities_gt[:,1], densities_gt[:,2], label="Samples (θ_gt)", color=:blue)
scatter!(p2, scaff_only_densities_gt[:,1], scaff_only_densities_gt[:,2], marker=:x, label="Scaffold (θ_gt)", color=:blue)

display(p2)

# Plot densities for θs_first_round
p3 = plot(xs, er_curve, color="gray", label="E-R", legend=:topleft, title="Densities for θs_first_round")
plot!(p3, xs, er_upper, color="gray", linestyle=:dash, label=false)
plot!(p3, xs_lower[2:end], er_lower[2:end], color="gray", linestyle=:dash, label=false)
scatter!(p3, densities[:,1], densities[:,2], label="Samples (θs_first_round)", color=:red)
scatter!(p3, scaff_only_densities[:,1], scaff_only_densities[:,2], marker=:x, label="Scaffold (θs_first_round)", color=:red)

display(p3)

# Plot densities for θ_gt
p4 = plot(xs, er_curve, color="gray", label="E-R", legend=:topleft, title="Densities for θ_gt")
plot!(p4, xs, er_upper, color="gray", linestyle=:dash, label=false)
plot!(p4, xs_lower[2:end], er_lower[2:end], color="gray", linestyle=:dash, label=false)
scatter!(p4, densities_gt[:,1], densities_gt[:,2], label="Samples (θ_gt)", color=:blue)
scatter!(p4, scaff_only_densities_gt[:,1], scaff_only_densities_gt[:,2], marker=:x, label="Scaffold (θ_gt)", color=:blue)

display(p4)


learning_rates = [0.1]
regularization_terms = [1e-3]

for lr in learning_rates
    for reg in regularization_terms
        println("Testing learning rate: $lr with regularization: $reg")
        
        set_parameters(model, [0.0, 0.0])
        θs_hessian = monte_carlo_gradient_ascent_hessian(
            model, Es, Dict(:burn_in => burn_in, :sample_interval => sample_interval), 100, iterations, lr, reg  # Regularization term added
        )

        println(θs_hessian)

        # Plot the parameter updates
        p = plot(1:iterations, θs_hessian[:, 1], label="Estimated Parameter 1 (LR = $lr, Reg = $reg)", xlabel="Iteration", ylabel="Parameter Value", title="Parameter Convergence with Normalized Hessian and Learning Rate $lr, Regularization $reg")
        plot!(p, 1:iterations, θs_hessian[:, 2], label="Estimated Parameter 2 (LR = $lr, Reg = $reg)")

        # Plot true parameters as horizontal lines
        hline!(p, [θ_gt[1]], label="True Parameter 1", linestyle=:dot, color=:red)
        hline!(p, [θ_gt[2]], label="True Parameter 2", linestyle=:dot, color=:blue)

        display(p)
    end
end



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


