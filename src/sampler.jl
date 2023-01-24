module sampler

import StatsBase
using ergm.models
using ergm.spaces
using ergm.stats

export GibbsSampler, sample, ParallelGibbsSampler, update_sampler_params, sample_stats

struct GibbsSampler
    model :: ERGM
    state
    burn_in
    sample_interval
end

function GibbsSampler(model :: ERGM; burn_in, sample_interval)
    state = empty(model.sample_space)
    set_state(model.stats, state)
    GibbsSampler(model, state, burn_in, sample_interval)
end

"""
Take on step along the Markov chain underlying the Gibbs sampler.
At most one edge will be toggled during this step.
"""
function gibbs_step(s :: GibbsSampler)
    # choose edge uniformly and propose toggling it
    index = random_index(s.model.sample_space)
    odds = conditional_log_odds(s.model, index)

    # sample new value from model distribution conditioned
    # on all indices but i
    if rand() < 1 / (1 + exp(-odds))
        s.state[index] = 1
        apply_update(s.model.stats, (index, 1))
    else
        s.state[index] = 0
        apply_update(s.model.stats, (index, 0))
    end
end

"""
Draw num_samples approximately independent samples from the model.

Returns
-------
sample_graphs: Vector of approximately independent graphs sampled from the model.
sample_stats: Matrix of sufficient statistics for each sample. The sufficient
    statistics of sample_graphs[i] are the vector sample_stats[:, i].
"""
function sample(s :: GibbsSampler, num_samples)
    num_params = stat_count(s.model.stats)
    sample_graphs = []
    sample_stats = zeros(num_params, num_samples)
    
    # burn in to reach equilibrium state
    for _ ∈ 1:s.burn_in
        gibbs_step(s)
    end

    for i ∈ 1:num_samples
        # draw one sample
        push!(sample_graphs, copy(s.state))
        sample_stats[:, i] = get_stats(s.model.stats)
        
        # throw away some samples to reduce autocorrelation
        for _ ∈ s.sample_interval
            gibbs_step(s)
        end
    end
    
    sample_graphs, sample_stats
end

struct ParallelGibbsSampler
    n_samplers
    samplers :: Vector{GibbsSampler}

    function ParallelGibbsSampler(model :: ERGM, burn_in, sample_interval, n_samplers)
        samplers = [
            GibbsSampler(copy(model), burn_in, sample_interval)
            for _ ∈ 1:n_samplers
        ]
        new(n_samplers, samplers)
    end
end

function sample(sampler :: ParallelGibbsSampler, n)
    # split requested samples evenly across chains
    m, r = divrem(n, sampler.n_samplers)
    n_per_chain = vcat(
        fill(m + 1, r),
        fill(m, sampler.n_samplers - r)
    )
    hs = []

    for i ∈ 1:sampler.n_samplers
        h = Threads.@spawn sample(sampler.samplers[i], n_per_chain[i])
        push!(hs, h)
    end

    np = length(sampler.samplers[1].params)
    sample_graphs = []
    sample_stats = zeros(np, n)
    j = 1

    for (i, h) ∈ enumerate(hs)
        Gs, ss = fetch(h)
        append!(sample_graphs, Gs)
        sample_stats[:, j:j + n_per_chain[i] - 1] = ss
        j += n_per_chain[i]
    end

    sample_graphs, sample_stats
end

end
