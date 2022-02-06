module stats

export SimpleStats, set_state, update_state, get_stats

mutable struct SimpleStats
    stats_function
    state

    function SimpleStats(stats_function)
        new(stats_function, undef)
    end
end

function set_state(stats :: SimpleStats, state)
    stats.state = state
end

function update_state(stats :: SimpleStats, update)
    i, x = update
    stats.state[i] = x
end

function get_stats(stats :: SimpleStats)
    stats.stats_function(stats.state)
end

function get_stats(stats :: SimpleStats, state)
    stats.stats_function(state)
end

end
