module spaces

import Base.copy, Base.getindex, Base.setindex!, Base.keys
import Graphs

export DiGraph, getdomain

struct DiGraph
    adjacency
    n
    
    function DiGraph(adjacency :: AbstractMatrix{Bool})
        m, n = size(adjacency)
        
        if m != n
            error("adjacency matrix must be square")
        end

        new(adjacency, n)
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
    i, j = p
    g.adjacency[i, j] = v
end

function Base.copy(g :: DiGraph)
  DiGraph(copy(g.adjacency))
end

end
