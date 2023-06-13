import Base.copy, Base.getindex, Base.setindex!, Base.empty
import StatsBase
using SparseArrays
using LinearAlgebra

export SparseGraphs, SparseGraph, random_index

"""
Directed graphs backed by sparse matrices.

This type includes a parameter `n` to specify the number of nodes. These graphs also have no self loops and
edges are binary. To create an empty graph on `n` nodes, use `SparseDirectedGraph{n}()`. To create
a graph from a sparse adjacency matrix `adjacency`, use `SparseDirectedGraph(adjacency)`.

Edge indices for this type take the form `(s, t)` where `s::Int` and `t::Int` are the indices of
the respective source and target nodes of the edge. Node indices are integers `i` in the range `1 ≤ i ≤ n`, where `n`
is the type paramter specifying the number of nodes in the graph. For example, querying the value of the edge from node
`1` to node `3` in the graph `x :: SparseDirectedGraph` is accomplished by `getindex(x, (1, 3))` or, equivalently, `x[(1, 3)]`.
"""
mutable struct SparseDirectedGraph{n} <: SampleSpace
    adjacency::SparseMatrixCSC{Bool,Int}
    updates_since_dropzeros::Int
    
    """
        SparseDirectedGraph{n}()

    Initialize empty sparse graph.
    """
    function SparseDirectedGraph{n}() where n
        adjacency = spzeros(Bool, n, n)
        new{n}(adjacency, 0)
    end
    
    """
        SparseDirectedGraph(adjacency)

    Initialize sparse graph from sparse adjacency matrix.

    Throws an error if `adjacency` can't be converted to `SparseMatrixCSC{Bool,Int}`
    """
    function SparseDirectedGraph(adjacency)
        _adjacency = SparseMatrixCSC{Bool, Int}(adjacency)
        r, c = size(_adjacency)
        
        if r != c
            error("Adjacency matrix must be square.")
        end

        # clear diagonal (no self-loops allowed)
        adjacency[diagind(_adjacency)] .= false

        new{r}(_adjacency, 0)
    end
end

function random_index(g::SparseDirectedGraph{n}) where n
    index = StatsBase.sample(1:n, 2, replace=false)
    tuple(index...)
end

function Base.getindex(g::SparseDirectedGraph, index::Tuple{Int,Int})::Bool
    i, j = index

    if i == j
        error("Cannot get diagonal elements in a SparseDirectedGraph.")
    end

    g.adjacency[i, j]
end

function Base.setindex!(g::SparseDirectedGraph, value::Bool, index::Tuple{Int,Int})
    i, j = index

    if i == j
        error("Cannot set diagonal elements in a SparseDirectedGraph.")
    end

    # don't insert zeros unnecessarily
    if value == 0 && g[index] == 0
        return value
    end

    # occasionally drop explicit zeros
    if g.updates_since_dropzeros > 500
        dropzeros!(g.adjacency)
        g.updates_since_dropzeros = 0
    end

    g.updates_since_dropzeros += 1
    g.adjacency[i, j] = value
end

function Base.copy(g::SparseDirectedGraph)
    SparseDirectedGraph(copy(g.adjacency))
end
