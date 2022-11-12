using ergm.spaces
using SparseArrays
using LinearAlgebra

mutable struct EdgeTriangle
    graph :: SparseGraph
    edge_count
    triangle_count
    max_edge_count
    max_triangle_count
    cached_counts
    
    function EdgeTriangle(space)
        if !(space isa SparseGraphs)
            print("EdgeTriangle statistics only supported \
                  for undirected sparse graphs.")
        end

        n = space.number_of_nodes

        new(
            empty(space),
            0,
            0,
            n * (n - 1) ÷ 2,
            n * (n - 1) * (n - 2) ÷ 6,
            Dict()
        )
    end
end

function stat_count(s :: EdgeTriangle)
    2
end

function set_state(s :: EdgeTriangle, graph :: SparseGraph)
    s.graph = copy(graph)
    A = graph.adjacency
    s.edge_count = sum(A) ÷ 2
    s.triangle_count = sum(A ^ 2 .* A) ÷ 6
    s.cached_counts = Dict()
end

function test_update(s :: EdgeTriangle, update; counts=false)
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
        A = s.graph.adjacency
        new_edge_count = s.edge_count + x - old_x
        new_triangle_count = s.triangle_count + (x - old_x) * sum(A[i, :] .* A[:, j])
        [new_edge_count, new_triangle_count]
    end
    
    s.cached_counts[update] = new_counts
    
    if counts
        new_counts
    else
        new_counts ./ [s.max_edge_count, s.max_triangle_count]
    end
end

function apply_update(s :: EdgeTriangle, update)
    s.edge_count, s.triangle_count = test_update(s, update; counts=true)
    ix, x = update
    s.graph[ix] = x
    s.cached_counts = Dict()
end

function get_stats(s :: EdgeTriangle)
    [s.edge_count, s.triangle_count] ./ [s.max_edge_count, s.max_triangle_count]
end

function get_stats(s :: EdgeTriangle, G :: SparseGraph)
    set_state(s, G)
    get_stats(s)
end
