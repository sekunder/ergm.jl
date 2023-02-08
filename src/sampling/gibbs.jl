using ergm.spaces
using ergm.models
using Random
import StatsBase
using Statistics
using GLMakie

struct GibbsSampler
    model::Model
    burn_in::Int
    sample_interval::Int

    @doc """
        GibbsSampler(model::Model; burn_in::Int, sample_interval::Int)

    Construct a Gibbs sampler for an given ERGM.

    A Gibbs sampler implements a Markov chain where each step can
    toggle at most one edge in the state graph. Before returning
    the first sample, `burn_in` steps are taken along the Markov chain.
    After the first sample, `sample_interval` steps are taken between
    each pair of samples returned with the goal of producing approximately
    independent samples.
    """
    function GibbsSampler(model::Model; burn_in::Int, sample_interval::Int)
        new(model, burn_in, sample_interval)
    end
end

"""
Take one step along the sampler's Markov chain.

Each time `gibbs_step` is invoked, at most one edge in the
chain's underlying state graph can be modified.
"""
function gibbs_step(sampler::GibbsSampler)
    # uniformly choose an edge and propose toggling it (i.e., changing it to the value x)
    i = random_index(get_state(sampler.model))
    x = !get_state(sampler.model)[i]
    
    # compute acceptance probability of toggle
    δstats = test_update(sampler.model, i, x)
    θ = get_parameters(sampler.model)
    δlog_likelihood = sum(θ .* δstats)
    α = min(1, exp(δlog_likelihood))
    
    # perform toggle with acceptance probability α
    if rand() < α
        apply_update(sampler.model, i, x)
    end
end

"""
Draw `number_of_samples` approximately independent samples from this ERGM.

Returns the pair `(samples, statistics)` where `samples` is a vector of
graphs sampled from the ERGM and `statistics` is a matrix such that
`statistics[i, :]` is a vector of the sufficient statistics of the
sampled graph `samples[i]`.
"""
function sample(sampler::GibbsSampler, number_of_samples::Int)
    # number of model statistics/parameters
    p = length(get_parameters(sampler.model))
    
    samples = []
    statistics = zeros(number_of_samples, p)

    for i ∈ 1:number_of_samples
        to_skip = if i == 1
            sampler.burn_in
        else
            sampler.sample_interval
        end
        
        # perform initial burn in / throw away steps between samples
        for _ ∈ 1:to_skip
            gibbs_step(sampler)
        end

        push!(samples, copy(get_state(sampler.model)))
        statistics[i, :] = get_statistics(sampler.model)
    end  
    
    samples, statistics
end

"""
Plot diagnostic functions useful for choosing Gibbs sampler parameters.

This function takes `steps` steps along the Gibbs sampler Markov chain
starting from an empty graph and records the value of the graph statistics
at every individual step. This is used to compute rolling means of each statistics
with window size `window_size` as well as the autocorrelation function
of each of the statistics.

Typically, we want to choose a burn-in length for the sampler long enough for
all the rolling means of the statistics to stabilize, which can be judged
heuristically just by looking at the plot of the rolling means.

We also need to choose an interval between samples long enough that the
autocorrelation of each of the statistics drops off near zero between
returned samples, which is a necessary condition for the sampler to
generate high-equality, approximately independent samples.

This function returns a matrix of all graph statistics, a matrix of
graph statistics smoothed with the given window size, a matrix of
autocorrelation functions, and the plot object.
"""
function plot_diagnostics(sampler::GibbsSampler, steps::Int; window_size=1000)
    # reset model state to empty graph
    S = get_sample_space(sampler.model)
    set_state(sampler.model, S())

    p = length(get_parameters(sampler.model))
    statistics = zeros(steps, p)

    for i ∈ 1:steps
        gibbs_step(sampler)
        statistics[i, :] = get_statistics(sampler.model)
    end

    # compute rolling means for each statistic
    smoothed_statistics = zeros(steps, p)

    for i ∈ 1:steps
        w = window_size ÷ 2
        a = max(1, i - w)
        b = min(i + w, steps)
        smoothed_statistics[i, :] = mean(
            statistics[a:b], dims=1
        )[1, :]
    end

    # compute autocorrelation for each statistic
    autocorrelation = StatsBase.autocor(statistics, 1:steps - 1)

    figure = Figure()
    mean_ax = Axis(figure[1, 1])
    cor_ax = Axis(figure[1, 2])

    Label(figure[1, 1, Top()], "Smoothed Statistics (for burn-in)")
    Label(figure[1, 2, Top()], "Autocorrelation of Statistics (for sample interval)")

    for i ∈ 1:p
        lines!(mean_ax, smoothed_statistics[:, i])
        lines!(cor_ax, autocorrelation[:, i])
    end

    display(figure)
    statistics, smoothed_statistics, autocorrelation, figure
end