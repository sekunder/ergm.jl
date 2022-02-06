module inference

using ergm.sampler, ergm.models
using Statistics
import StatsBase
export mcmc_mle

function mcmc_estimate_means(
    sampler_initial_state,
    model :: ExponentialFamily,
    estimation_steps,
    burn_in,
    sample_interval
    )
    # seed MC with one of our samples (may not be the best choice...)
    s = GibbsSampler(
        sampler_initial_state,
        model,
        burn_in,
        sample_interval
    )
    _, stats = sample(s, estimation_steps)
    mean_stats = mean(stats, dims=1)
    mean_stats
end

function mcmc_mle(
    observations, model :: ExponentialFamily,
    gradient_descent_steps, estimation_steps, learning_rate
    )

    α = learning_rate
    θ = initial_guess
    θs = [θ]
    Ls = []

    observation_stats = [get_stats(model.stats, o) for o ∈ observations]
    target_Es = mean(reduce(hcat, observation_stats), dims=2)

    # need better heuristic
    burn_in = 10
    sample_interval = 10

    for i ∈ 1:gradient_descent_steps
        # seed sampler with random observation
        G0 = StatsBase.sample(observations)
        Es = mcmc_estimate_means(
            G0,
            model,
            estimation_steps,
            burn_in,
            sample_interval
        )
        dθ = target_Es - Es
        θ .+= α * dθ
        L = undef
        push!(Ls, L)
    end

    Ls
end

end
