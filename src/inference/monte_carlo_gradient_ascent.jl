using ergm.models
using ergm.sampling
using Statistics

function monte_carlo_gradient_ascent(model::Model, target_statistics::Vector{Float64}, sampler_parameters::Dict, gradient_samples::Int, fitting_iterations::Int, learning_rate::Float64)
    sampler = GibbsSampler(model; sampler_parameters...)
    p = length(get_parameters(model))
    θ = zeros(p)
    θs = zeros(fitting_iterations, p)
    
    @showprogress for i ∈ 1:fitting_iterations
        _, ss = sample(sampler, gradient_samples)
        current_statistics = mean(ss, dims=1)[1, :]
        ∇log_likelihood = target_statistics - current_statistics;
        θ += learning_rate * ∇log_likelihood
        set_parameters(model, θ)
        θs[i, :] = θ
    end

    θs
end
