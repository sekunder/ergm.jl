module sampler

import StatsBase
using ergm.spaces

export GibbsSampler, sample, SimpleLikelihood

mutable struct GibbsSampler
    initial_state
    state
    likelihood
    burn_in
    sample_interval
    
    function GibbsSampler(initial_state, likelihood, burn_in, sample_interval)
        sampler = new(initial_state, undef, likelihood, burn_in, sample_interval)
        restart(sampler)
        sampler
    end
end

function restart(sampler :: GibbsSampler)
    sampler.state = copy(sampler.initial_state)
    set(sampler.likelihood, sampler.state)
end

function gibbs_step(sampler :: GibbsSampler)
    is = keys(sampler.state)
    n = length(is)

    for i ∈ is
        w = zeros(n)
        d = getdomain(sampler.state, i)

        for (j, x) ∈ enumerate(d)
            sampler.state[i] = x
            w[j] = get(sampler.likelihood, i, x)
        end

        x = StatsBase.sample(d, StatsBase.Weights(w))
        sampler.state[i] = x
        update(sampler.likelihood, i, x)
    end
end

function sample(sampler :: GibbsSampler, n)
    restart(sampler)
    samples = []
    
    # burn in to reach equilibrium state
    for _ ∈ 1:sampler.burn_in
        gibbs_step(sampler)
    end

    for _ ∈ 1:n
        # draw one sample
        push!(samples, copy(sampler.state))
        
        # throw away some samples to reduce autocorrelation
        for _ ∈ sampler.sample_interval
            gibbs_step(sampler)
        end
    end
    
    samples
end

mutable struct SimpleLikelihood
    likelihood_function
    state
  
    function SimpleLikelihood(likelihood_function)
      new(likelihood_function, undef)
    end
end

function set(l :: SimpleLikelihood, state)
    l.state = copy(state)
end

function get(l :: SimpleLikelihood, i, x)
    old_x = l.state[i]
    l.state[i] = x
    lv = l.likelihood_function(l.state)
    l.state[i] = old_x
    lv
end

function update(l :: SimpleLikelihood, i, x)
    l.state[i] = x
end

end
