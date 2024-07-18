using ergm.models
using ergm.sampling
using Statistics
using LinearAlgebra

function monte_carlo_gradient_ascent(model::Model,
    target_statistics::Vector{Float64}, sampler_parameters::Dict, 
    gradient_samples::Int, fitting_iterations::Int, learning_rate::Float64)
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

function monte_carlo_gradient_ascent_hessian(model::Model, target_statistics::Vector{Float64}, sampler_parameters::Dict, gradient_samples::Int, fitting_iterations::Int, learning_rate::Float64, regularization::Float64 = 1e-5)
    sampler = GibbsSampler(model; sampler_parameters...)
    p = length(get_parameters(model))
    θ = zeros(p)
    θs = zeros(fitting_iterations, p)

    function gradient_and_hessian(θ)
        set_parameters(model, θ)
        _, ss = sample(sampler, gradient_samples)
        current_statistics = mean(ss, dims=1)[1, :]
        grad = target_statistics - current_statistics

        H = cov(ss, dims=1) + regularization * I(p)  # Regularized Hessian
        return grad, H
    end
    
    @showprogress for i ∈ 1:fitting_iterations
        ∇log_likelihood, H = gradient_and_hessian(θ)
        if det(H) < 1e-10  # Check for near-singular matrices
            println("Hessian is nearly singular at iteration $i, determinant: $(det(H))")
            H += regularization * I(p)  # Further regularize if nearly singular
        end
        H_inv = inv(H)
        θ += learning_rate * H_inv * ∇log_likelihood / norm(∇log_likelihood) 
        set_parameters(model, θ)
        θs[i, :] = θ
    end

    θs
end

