module inference

using ergm.sampler, ergm.models, ergm.stats
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
    s = ParallelGibbsSampler(
        sampler_initial_state,
        model,
        burn_in,
        sample_interval,
        Threads.nthreads()
    )
    _, stats = sample(s, estimation_steps)
    mean_stats = mean(stats, dims=1)
    mean_stats
end

function mcmc_mle(
    observations, model :: ExponentialFamily,
    gradient_descent_steps,
    estimation_steps, burn_in, sample_interval,
    learning_rate,
    )

    α = learning_rate
    θ = get_params(model)
    m = length(θ)
    θs = zeros(gradient_descent_steps + 1, m)
    θs[1, :] = θ
    Ls = zeros(gradient_descent_steps)

    observation_stats = [get_stats(model.stats, o) for o ∈ observations]
    target_Es = mean(reduce(hcat, observation_stats), dims=2)

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
        θs[i + 1, :] = θ
        L = sum(dθ .^ 2)
        Ls[i] = L
    end

    θs, Ls
end

end
