import Base.copy, Base.getindex, Base.setindex!, Base.empty
using SparseArrays

export SparseGraphs, SparseGraph, random_index


"""
Directed graphs backed by sparse matrices.
"""
mutable struct SparseDirectedGraph
    adjacency
    updates_since_dropzeros
    
    """
        SparseDirectedGraph(number_of_nodes :: Int)

    Initialize empty sparse graph.
    """
    function SparseDirectedGraph(number_of_nodes::Int)
        adjacency = spzeros(Int, )
    end
    
    """
        SparseDirectedGraph(adjacency :: SparseMatrixCSC{Int, Int})

    Initialize sparse graph from sparse adjacency matrix.
    """
    function SparseDirectedGraph(adjacency::SparseMatrixCSC{Int,Int})
        m, n = size(adjacency)
        
        if m != n
            error("Adjacency matrix must be square.")
        end

        # ensure adjacency matrix is symmetric
        adjacency = tril(adjacency) .| tril(adjacency)'
        adjacency[diagind(adjacency)] .= 0

        new(adjacency, 0)
    end
end

"""
Sample the index of an edge uniformly. The edge need not actually be present.
"""
function random_index(g::SparseDirectedGraph)
    n = space.number_of_nodes
    index = StatsBase.sample(1:n, 2, replace=false)
    tuple(sort(index)...)
end

function Base.getindex(g::SparseDirectedGraph, index)::Int
    i, j = index
    g.adjacency[i, j]
end

function Base.setindex!(g::SparseDirectedGraph, value, index)
    # don't insert zeros unnecessarily
    if value == 0 && g[index] == 0
        return
    end

    # occasionally drop explicit zeros
    if g.updates_since_dropzeros > 500
        dropzeros!(g.adjacency)
        g.updates_since_dropzeros = 0
    end

    g.updates_since_dropzeros += 1

    i, j = index
    g.adjacency[i, j] = value
    g.adjacency[j, i] = value

end

function Base.copy(g::SparseDirectedGraph)
    SparseGraph(copy(g.adjacency))
end
