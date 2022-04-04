module stats

export SimpleStats, set_state, update_state, get_stats, DeltaStats, test_state, edge_stats, directed_triplet_stats
using LinearAlgebra

mutable struct SimpleStats
    stats_function
    state

    function SimpleStats(stats_function)
        new(stats_function, undef)
    end
end

function set_state(stats :: SimpleStats, state)
    stats.state = copy(state)
end

function update_state(stats :: SimpleStats, update)
    i, x = update
    stats.state[i] = x
end

function test_state(stats :: SimpleStats, update)
    i, x = update
    old_x = stats.state[i]
    stats.state[i] = x
    test_stats = stats.stats_function(stats.state)
    stats.state[i] = old_x
    test_stats
end

function get_stats(stats :: SimpleStats)
    stats.stats_function(stats.state)
end

function get_stats(stats :: SimpleStats, state)
    stats.stats_function(state)
end

function Base.copy(stats :: SimpleStats)
    SimpleStats(stats.stats_function)
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
    stats.state = copy(state)
    stats.current_stats = stats.stats_function(state)
end

function update_state(stats :: DeltaStats, update)
    i, x = update
    stats.current_stats = stats.delta_stats_function(
        stats.state,
        stats.current_stats,
        update
    )
    stats.state[i] = x
end

function test_state(stats :: DeltaStats, update)
    i, x = update
    stats.delta_stats_function(
        stats.state,
        stats.current_stats,
        update
    )
end

function get_stats(stats :: DeltaStats)
    stats.current_stats
end

function get_stats(stats :: DeltaStats, state)
    stats.stats_function(state)
end

function Base.copy(stats :: DeltaStats)
    DeltaStats(stats.stats_function, stats.delta_stats_function)
end

s_string = [
"122223333444456",  # 0
"010000011111223",  # 1
"001001311232246",  # 2
"000101010211123",  # 3
"000011001112123",  # 4
"000001000212036",  # 5
"000000100010012",  # 6
"000000010210236",  # 7
"000000001012236",  # 8
"000000000100013",  # 9
"000000000010026",  # 10
"000000000001013",  # 11
"000000000000113",  # 12
"000000000000016",  # 13
"000000000000001"]
teto = vcat([[parse(Integer, c) for c ∈ r]' for r ∈ s_string]...)
tote = round.(Integer, teto^-1)

directed_triplet_stats = DeltaStats(
    function(G)
        s = zeros(Integer, 15)
        A = G.adjacency
        n, _ = size(A)
        m = sum(A)
        g_sq = A^2
        g_cu = g_sq * A
        tr_g_sq = tr(g_sq)
        s[1] = (n - 2) * m
        s[2] = (n - 2) * tr_g_sq ÷ 2
        s[3] = sum(g_sq) - tr_g_sq
        div = A' * A
        conv = A * A'
        s[4] = (sum(div) - m) ÷ 2
        s[5] = (sum(conv) - m) ÷ 2
        s[6] = sum(A .* g_sq)
        s[7] = tr(g_cu) ÷ 3
        g_sym = A .* A'
        bidegi = sum(g_sym, dims=2)
        s[8] = sum(A .* bidegi) - sum(g_sym)
        s[9] = sum(A .* bidegi') - sum(g_sym)
        s[10] = sum(g_sym .* conv) ÷ 2
        s[11] = sum(g_sym .* g_sq) 
        s[12] = sum(g_sym .* div) ÷ 2
        g_sym_sq = g_sym^2
        s[13] = (sum(g_sym_sq) - tr(g_sym_sq)) ÷ 2
        s[14] = sum(g_sym_sq .* A)
        s[15] = tr(g_sym_sq * g_sym) ÷ 6
        c = tote * s
        c ./ (n * (n - 1) * (n - 2))
    end,
    function(G, s, up)
        A = G.adjacency
        n, _ = size(A)
        i, x = up
        u, v = i

        if A[u, v] == x
            return s
        end

        u, v = i
        Duv = 2 * A[u, v] - 1
        δ = zeros(Integer, 15)
        δ[1] = n - 2
        δ[2] = (n - 2) * A[v, u]
        outu = sum(A[u, :])
        inu = sum(A[:, u])
        outv = sum(A[v, :])
        inv = sum(A[:, v])
        δ[3] = inu + outv - 2 * A[v, u]
        δ[4] = outu - A[u, v]
        δ[5] = inv - A[u, v]

        common_post = A[u, :] .* A[v, :]
        common_pre = A[:, u] .* A[:, v]
        δ[6] = sum(common_post) + sum(common_pre) + dot(A[u, :], A[:, v])
        δ[7] = dot(A[v, :], A[:, u])

        bidegu = dot(A[u, :], A[:, u])
        bidegv = dot(A[v, :], A[:, v])
        δ[8] = bidegu + A[v, u] * (outu + outv - 2 * A[u, v] - 2 * A[v, u] + 1)
        δ[9] = bidegv + A[v, u] * (inu + inv - 2 * A[u, v] - 2 * A[v, u] + 1)
        
        δ[10] = dot(A[u, :], common_pre) + A[v, u] * sum(common_post)
        δ[11] = dot(common_post, A[:, u]) + dot(A[v, :], common_pre)
        δ[12] = dot(common_post, A[:, v]) + A[v, u] * sum(common_pre)

        δ[13] = A[v, u] * (bidegu + bidegv - 2 * A[u, v])
        δ[14] = dot(common_post, common_pre)

        if A[v, u] != 0
            δ[11] += dot(A[u, :], A[:, v]) + dot(A[v, :], A[:, u])
            δ[14] += dot(common_post, A[:, v]) + dot(common_post, A[:, u]) +
                     dot(A[u, :], common_pre) + dot(A[v, :], common_pre)
            δ[15] = dot(common_post, common_pre)
        end

        δ = tote * δ
        s - Duv * δ ./ (n * (n - 1) * (n - 2))
    end
)

end
