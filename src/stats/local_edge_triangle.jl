using ergm.spaces
using SparseArrays
using LinearAlgebra
import Base

mutable struct LocalEdgeTriangle
    graph :: SparseGraph
    local_graph :: SparseGraph
    edge_count
    triangle_count
    max_edge_count
    max_triangle_count
    cached_counts
    node_positions
    local_radius
end
    
function LocalEdgeTriangle(space, node_positions, local_radius)
    if !(space isa SparseGraphs)
        print("LocalEdgeTriangle statistics only supported \
              for undirected sparse graphs.")
    end

    n = space.number_of_nodes

    LocalEdgeTriangle(
        empty(space),
        empty(space),
        0,
        0,
        n, # n * (n - 1) ÷ 2,
        n, # n * (n - 1) * (n - 2) ÷ 6,
        Dict(),
        node_positions,
        local_radius
    )
end

function Base.copy(s :: LocalEdgeTriangle)
    LocalEdgeTriangle(
        copy(s.graph),
        copy(s.local_graph),
        s.edge_count,
        s.triangle_count,
        s.max_edge_count,
        s.max_triangle_count,
        copy(s.cached_counts),
        copy(s.node_positions),
        s.local_radius
    )
end

function stat_count(s :: LocalEdgeTriangle)
    2
end

function set_state(s :: LocalEdgeTriangle, graph :: SparseGraph)
    A = graph.adjacency
    i, j, v = findnz(A)

    # filter out non-local edges
    X = s.node_positions
    lix = [
        ix for ix ∈ 1:length(i) 
        if norm(X[i[ix], :] - X[j[ix], :]) < s.local_radius
    ]
    LA = sparse(i[lix], j[lix], v[lix], size(A)...)

    s.graph = copy(graph)
    s.local_graph = SparseGraph(LA)
    s.edge_count = sum(A) ÷ 2
    s.triangle_count = sum(LA ^ 2 .* LA) ÷ 6
    s.cached_counts = Dict()
end

function test_update(s :: LocalEdgeTriangle, update; counts=false)
    if update ∈ keys(s.cached_counts)
        if counts
            return s.cached_counts[update]
        else
            return s.cached_counts[update] ./ [s.max_edge_count, s.max_triangle_count]
        end
    end
    
    (i, j), x = update
    old_x = s.graph[(i, j)]
    
    new_counts = if x == old_x
        [s.edge_count, s.triangle_count]
    else
        LA = s.local_graph.adjacency
        X = s.node_positions
        l = norm(X[i, :] - X[j, :]) < s.local_radius
        new_edge_count = s.edge_count + (x - old_x)
        new_triangle_count = s.triangle_count + l * (x - old_x) * sum(LA[i, :] .* LA[:, j])
        [new_edge_count, new_triangle_count]
    end
    
    s.cached_counts[update] = new_counts
    
    if counts
        new_counts
    else
        new_counts ./ [s.max_edge_count, s.max_triangle_count]
    end
end

function apply_update(s :: LocalEdgeTriangle, update)
    s.edge_count, s.triangle_count = test_update(s, update; counts=true)
    (i, j), x = update
    s.graph[(i, j)] = x

    X = s.node_positions

    if norm(X[i, :] - X[j, :]) < s.local_radius
        s.local_graph[(i, j)] = x
    end

    s.cached_counts = Dict()
end

function get_stats(s :: LocalEdgeTriangle)
    [s.edge_count, s.triangle_count] ./ [s.max_edge_count, s.max_triangle_count]
end

function get_stats(s :: LocalEdgeTriangle, G :: SparseGraph)
    set_state(s, G)
    get_stats(s)
end
