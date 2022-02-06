module sampler

import StatsBase
using ergm.spaces, ergm.models
import ergm.stats

export GibbsSampler, sample

mutable struct GibbsSampler
    initial_state
    state
    model
    burn_in
    sample_interval
    
    function GibbsSampler(
            initial_state,
            model :: ExponentialFamily,
            burn_in,
            sample_interval
    )
        sampler = new(initial_state, undef, model, burn_in, sample_interval)
        restart(sampler)
        sampler
    end
end

function restart(sampler :: GibbsSampler)
    sampler.state = copy(sampler.initial_state)
    set_state(sampler.model, sampler.state)
end

function gibbs_step(sampler :: GibbsSampler)
    is = keys(sampler.state)
    n = length(is)

    for i ∈ is
        d = getdomain(sampler.state, i)
        w = zeros(length(d))
        old_x = sampler.state[i]

        for (j, x) ∈ enumerate(d)
            update_state(sampler.model, (i, x))
            w[j] = log_likelihood(sampler.model)
            update_state(sampler.model, (i, old_x))
        end

        # Try to avoid going outside Float64
        # precision when applying exp. Can
        # exponentiation be avoided here? Can
        # be avoided for length(d) == 2 but not
        # sure how to generalize.
        c = (maximum(w) + minimum(w)) / 2
        w = exp.(w .- c)
        x = StatsBase.sample(d, StatsBase.Weights(w))
        sampler.state[i] = x
        update_state(sampler.model, (i, x))
    end
end

function sample(sampler :: GibbsSampler, n)
    restart(sampler)
    samples = []
    m = length(sampler.model.params)
    sample_stats = zeros(n, m)
    
    # burn in to reach equilibrium state
    for _ ∈ 1:sampler.burn_in
        gibbs_step(sampler)
    end

    for i ∈ 1:n
        # draw one sample
        push!(samples, copy(sampler.state))
        sample_stats[i, :] = stats.get_stats(sampler.model.stats)
        
        # throw away some samples to reduce autocorrelation
        for _ ∈ sampler.sample_interval
            gibbs_step(sampler)
        end
    end
    
    samples, sample_stats
end

end
