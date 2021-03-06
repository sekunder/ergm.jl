module models

import ergm.stats

export ExponentialFamily, get_params, update_params, log_likelihood, set_state, update_state, test_state

mutable struct ExponentialFamily
    stats
    params

    function ExponentialFamily(stats, params)
        new(stats, params)
    end
end

function get_params(model :: ExponentialFamily)
    model.params
end

function update_params(model :: ExponentialFamily, params)
    model.params = params
end

function log_likelihood(model :: ExponentialFamily)
    sum(model.params .* stats.get_stats(model.stats))
end

function set_state(model :: ExponentialFamily, state)
    stats.set_state(model.stats, state)
end

function update_state(model :: ExponentialFamily, update)
    stats.update_state(model.stats, update)
end

function test_state(model :: ExponentialFamily, update)
    test_stats = stats.test_state(model.stats, update)
    sum(model.params .* test_stats)
end

function Base.copy(model :: ExponentialFamily)
    ExponentialFamily(copy(model.stats), copy(model.params))
end

end
