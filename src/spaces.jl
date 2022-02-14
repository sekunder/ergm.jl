module spaces

import Base.copy, Base.getindex, Base.setindex!, Base.keys
import Graphs
using SparseArrays
using LinearAlgebra

export DiGraph, getdomain

mutable struct DiGraph
    adjacency
    n
    updates_since_dropzeros
    
    function DiGraph(adjacency :: AbstractMatrix{Bool})
        m, n = size(adjacency)
        
        if m != n
            error("adjacency matrix must be square")
        end

        adjacency[diagind(adjacency)] .= 0
        new(adjacency, n, 0)
    end
end

function DiGraph(graph :: Graphs.DiGraph)
    A = Graphs.LinAlg.symmetrize(Graphs.LinAlg.adjacency_matrix(graph))
    A = Bool.(A)
    DiGraph(A)
end

Base.keys(g :: DiGraph) = [(i, j) for i in 1:g.n for j in 1:g.n if i != j]

function Base.getindex(g :: DiGraph, p)
  i, j = p
  g.adjacency[i, j]
end

getdomain(g :: DiGraph, i) = [false, true]

function Base.setindex!(g :: DiGraph, v, p)
    if typeof(g.adjacency) <: AbstractSparseMatrix
        # if sparse, don't insert zeros unnecessarily
        if v == 0 && g[p] == 0
            return
        end

        # if sparse, occasionally drop zeros
        if g.updates_since_dropzeros > 500
            dropzeros!(g.adjacency)
            g.updates_since_dropzeros = 0
        end

        g.updates_since_dropzeros += 1
    end

    i, j = p
    g.adjacency[i, j] = v

end

function Base.copy(g :: DiGraph)
  DiGraph(copy(g.adjacency))
end

end
