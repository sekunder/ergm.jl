using ergm.models
using ergm.sampling
using Statistics
using LinearAlgebra

function monte_carlo_gradient_ascent(model::Model, target_statistics::Vector{Float64}, sampler_parameters::Dict, gradient_samples::Int, fitting_iterations::Int, learning_rate::Float64)
    sampler = GibbsSampler(model; sampler_parameters...)
    p = length(get_parameters(model))
    θ = zeros(p)
    θs = zeros(fitting_iterations, p)
    
    @showprogress for i ∈ 1:fitting_iterations
        _, ss = sample(sampler, gradient_samples)
        current_statistics = mean(ss, dims=1)[1, :]
        ∇log_likelihood = target_statistics - current_statistics;
        θ += learning_rate * ∇log_likelihood / norm(∇log_likelihood)
        set_parameters(model, θ)
        θs[i, :] = θ
    end

    θs
end

function monte_carlo_gradient_ascent_hessian(model::Model, target_statistics::Vector{Float64}, sampler_parameters::Dict, gradient_samples::Int, fitting_iterations::Int, learning_rate::Float64)
    #epsilon = 1e-5 
    sampler = GibbsSampler(model; sampler_parameters...)
    p = length(get_parameters(model))
    θ = zeros(p)
    θs = zeros(fitting_iterations, p)

    function gradient_and_hessian(θ)
        set_parameters(model, θ)
        _, ss = sample(sampler, gradient_samples)
        current_statistics = mean(ss, dims=1)[1, :]
        grad = target_statistics - current_statistics

        # H = zeros(p, p)
        # for i in 1:p
        #     θ_i_plus = copy(θ)
        #     θ_i_minus = copy(θ)
        #     θ_i_plus[i] += epsilon
        #     θ_i_minus[i] -= epsilon

        #     set_parameters(model, θ_i_plus)
        #     _, ss_plus = sample(sampler, gradient_samples)
        #     current_statistics_plus = mean(ss_plus, dims=1)[1, :]
        #     grad_plus = target_statistics - current_statistics_plus

        #     set_parameters(model, θ_i_minus)
        #     _, ss_minus = sample(sampler, gradient_samples)
        #     current_statistics_minus = mean(ss_minus, dims=1)[1, :]
        #     grad_minus = target_statistics - current_statistics_minus

        #     H[:, i] = (grad_plus - grad_minus) / (2 * epsilon)
        # end
        H = cov(ss, dims = 1)
        return grad, H
    end
    
    @showprogress for i ∈ 1:fitting_iterations
        ∇log_likelihood, H = gradient_and_hessian(θ)
        H_inv = inv(H)
        θ -= H_inv * ∇log_likelihood  # Update rule using Hessian
        set_parameters(model, θ)
        θs[i, :] = θ
    end

    θs
end

