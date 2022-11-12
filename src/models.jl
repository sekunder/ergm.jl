module models

using ergm.spaces
using ergm.stats

export ERGM, log_likelihood, conditional_log_odds

"""
Exponential Random Graph Model.

sample_space: the space of graphs to consider, for example SparseGraph(50) for
sparse, undirected graphs on 50 nodes.

stats: object from ergm.stats for computing sufficient statistics and
updating them as graph edges are toggled

params: natural parameters for each statistic returned by stats
"""
mutable struct ERGM
    sample_space
    stats
    params :: Vector{Float64}

    function ERGM(sample_space, stats_type, params, covariates...)
        stats = stats_type(sample_space, covariates...)
        new(sample_space, stats, params)
    end
end

"""
Compute log likelihood (up to an additive constant).

Note that the model.stats object is stateful, and the
computed likelihood is for the current graph included
in the state of model.stats.
"""
function log_likelihood(model :: ERGM)
    sum(model.params .* get_stats(model.stats))
end

"""
Compute log odds conditioned on all but one edge.

In particular, compute the value:
log P(x_i = 1 | x_j for j ≠ i, θ) - log P(x_i = 0 | x_j for j ≠ i, θ),
where P denotes the model PMF and the values x_j for j ≠ i depend on the
state of model.stats.

index: index i that is not conditioned on
"""
function conditional_log_odds(model :: ERGM, index)
    δstats = test_update(model.stats, (index, 1)) - test_update(model.stats, (index, 0))
    sum(model.params .* δstats)
end

function Base.copy(model :: ERGM)
    ERGM(sample_space, copy(model.stats), copy(model.params))
end

end
