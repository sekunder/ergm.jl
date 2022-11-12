module stats

include("stats/edge_triangle.jl")
include("stats/local_edge_triangle.jl")

export stat_count, get_stats, set_state, test_update, apply_update
export EdgeTriangle
export LocalEdgeTriangle

end
