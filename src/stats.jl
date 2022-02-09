module stats

export SimpleStats, set_state, update_state, get_stats, DeltaStats

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

mutable struct DeltaStats
    stats_function
    delta_stats_function
    current_stats
    state

    function DeltaStats(stats_function, delta_stats_function)
        new(stats_function, delta_stats_function, undef, undef)
    end
end

function set_state(stats :: DeltaStats, state)
    stats.state = state
    stats.current_stats = stats.stats_function(state)
end

function update_state(stats :: DeltaStats, update)
    i, x = update
    stats.state[i] = x
    stats.current_stats = stats.delta_stats_function(
        stats.state,
        stats.current_stats,
        update
    )
end

function get_stats(stats :: DeltaStats)
    stats.current_stats
end

function get_stats(stats :: DeltaStats, state)
    stats.stats_function(stats.state)
end

end
