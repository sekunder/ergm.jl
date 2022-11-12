module spaces

import Base.copy, Base.getindex, Base.setindex!, Base.empty
import Graphs
import StatsBase
using SparseArrays
using LinearAlgebra

export SparseGraphs, SparseGraph, random_index

"""
Defines the space of undirected graphs on a fixed number
of nodes and backed by sparse graph representations.
"""
struct SparseGraphs
    number_of_nodes :: Int64
end

"""
Return an empty graph belonging to this space. That is,
an empty sparse undirected graph on n nodes.
"""
function Base.empty(space :: SparseGraphs)
    n = space.number_of_nodes
    SparseGraph(spzeros(Int64, n, n))
end

"""
Sample the index of an edge uniformly. The edge need not actually be present.
"""
function random_index(space :: SparseGraphs)
    n = space.number_of_nodes
    index = StatsBase.sample(1:n, 2, replace=false)
    tuple(sort(index)...)
end

"""
Defines one particular undirected sparse graph corresponding
to the SparseGraphs space defined above.
"""
mutable struct SparseGraph
    adjacency
    updates_since_dropzeros
    
    """
    Initialize sparse graph from sparse adjacency matrix.
    """
    function SparseGraph(adjacency :: SparseMatrixCSC{Int64, Int64})
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

function Base.getindex(g :: SparseGraph, index) :: Int64
  i, j = index
  g.adjacency[i, j]
end

function Base.setindex!(g :: SparseGraph, value, index)
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

function Base.copy(g :: SparseGraph)
  SparseGraph(copy(g.adjacency))
end

end
