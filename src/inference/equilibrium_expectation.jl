using ergm.models
using ergm.sampling
using ProgressMeter

function equilibrium_expectation(model::Model, target_statistics::Vector{Float64}, estimation_steps::Int, learning_rate::Vector{Float64})
    c1 = 1e-2
    p1 = 2
    p2 = 1/2
    fitting_iterations = length(learning_rate)
    sampler = GibbsSampler(model)
    p = length(get_parameters(model))
    D = ones(p)
    θ = get_parameters(model)
    θs = zeros(fitting_iterations + 1, p)
    θs[1, :] = θ
    Ds = zeros(fitting_iterations + 1, p)
    Ds[1, :] = D
    fs = nothing
    
    @showprogress for i ∈ 1:fitting_iterations
        c2 = learning_rate[i]

        for _ ∈ 1:estimation_steps
            gibbs_step(sampler)
        end
        
        δstats = get_statistics(model) - target_statistics
        θ -= D .* sign.(δstats) .* δstats .^ p1
        set_parameters(model, θ)
        θs[i + 1, :] = θ
        Ds[i + 1, :] = D

        # adapt learning rates every a iterations
        a = 500

        if i % a == 0
            ix = (i - a + 1):i
            θ_m = mean(θs[ix, :], dims=1)[1, :]
            θ_sd = std(θs[ix, :], dims=1)[1, :]
            f = (θ_sd ./ max.(abs.(θ_m), c2)) ./ c2
            D .*= (1 ./ f) .^ p2

            if fs === nothing
                fs = f'
            else
                fs = vcat(fs, f')
            end
        end
    end

    θs, Ds, fs
end
