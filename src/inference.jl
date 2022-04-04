module inference

using ergm.sampler, ergm.models, ergm.stats, ergm.optim
using Statistics
import StatsBase
export mcmc_mle, mcmc_mle_from_stats

function mcmc_mle(
    observations, model :: ExponentialFamily, optimizer,
    estimation_steps, burn_in, sample_interval)
    observation_stats = hcat([get_stats(model.stats, o) for o ∈ observations]...)
    target_Es = mean(observation_stats, dims=2)[:, 1]
    # seed sampler with observations
    mcmc_mle_from_stats(target_Es, observations, model, optimizer, estimation_steps, burn_in, sample_interval)
end

function mcmc_mle_from_stats(
    target_Es, sampler_seeds, model :: ExponentialFamily, optimizer,
    estimation_steps, burn_in, sample_interval)

    m = length(get_params(model))
    θs = []
    Ls = []
    Es = []

    while !done(optimizer)
        G0 = StatsBase.sample(sampler_seeds)
        s = ParallelGibbsSampler(
            G0,
            model,
            burn_in,
            sample_interval,
            Threads.nthreads()
        )
        _, ss = sample(s, estimation_steps)
        push!(Es, mean(ss, dims=2))
        θ = optim_step(optimizer, ss, target_Es)
        update_params(model, θ)
        push!(θs, θ)
        g = mean(ss, dims=2) - target_Es
        L = sum(g .^ 2)
        push!(Ls, L)
    end

    hcat(θs...), Ls, target_Es, Es
end

function ee(
    observations, model :: ExponentialFamily,
    fitting_steps, estimation_steps, learning_rate)
end

end
