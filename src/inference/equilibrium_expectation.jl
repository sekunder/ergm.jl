using ergm.models
using ergm.sampling
using ProgressMeter

function equilibrium_expectation(model::Model, target_statistics::Vector{Float64}, estimation_steps::Int, fitting_iterations::Int, learning_rate::Float64)
    c1 = 1e-2
    p1 = 2
    p2 = 1/2

    sampler = GibbsSampler(model)
    p = length(get_parameters(model))
    D = ones(p)
    θ = zeros(p)
    θs = zeros(fitting_iterations, p)
    
    @showprogress for i ∈ 1:fitting_iterations
        for _ ∈ 1:estimation_steps
            gibbs_step(sampler)
        end
        
        δstats = get_statistics(model) - target_statistics
        θ -= D .* sign.(δstats) .* δstats .^ p1
        θs[i, :] = θ

        # adapt learning rates every a iterations
        a = 500

        if it % a == 0
            ix = (i - a + 1):i
            θ_m = mean(θs[ix, :], dims=1)[1, :]
            θ_sd = std(θs[ix, :], dims=1)[1, :]
            D .*= (c2 * max.(abs.(θ_m), c1) ./ θ_sd) .^ p2
        end
    end

    θs
end