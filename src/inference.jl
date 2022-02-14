module inference

using ergm.sampler, ergm.models, ergm.stats, ergm.optim
using Statistics
import StatsBase
export mcmc_mle, mcmc_estimate_means

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
    mean_stats = mean(stats, dims=1)'
    mean_stats
end

function mcmc_mle(
    observations, model :: ExponentialFamily, optimizer,
    estimation_steps, burn_in, sample_interval,
    )

    m = length(get_params(model))
    θs = []
    Ls = []

    observation_stats = hcat([get_stats(model.stats, o) for o ∈ observations]...)
    target_Es = mean(observation_stats, dims=2)

    while !done(optimizer)
        # seed sampler with random observation
        G0 = StatsBase.sample(observations)
        Es = mcmc_estimate_means(
            G0,
            model,
            estimation_steps,
            burn_in,
            sample_interval
        )
        dθ = Es .- target_Es
        θ = optim_step(optimizer, dθ)
        update_params(model, θ)
        push!(θs, θ)
        L = sum(dθ .^ 2)
        push!(Ls, L)
    end

    hcat(θs...), Ls
end

end
