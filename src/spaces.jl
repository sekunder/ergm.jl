module spaces

import Base.copy, Base.getindex, Base.setindex!, Base.keys

export DenseDiGraph, getdomain

struct DenseDiGraph
    adjacency
    n
    
    function DenseDiGraph(adjacency :: Matrix{Bool})
        m, n = size(adjacency)
        
        if m != n
            error("adjacency matrix must be square")
        end

        new(adjacency, n)
    end
end

Base.keys(g :: DenseDiGraph) = [(i, j) for i in 1:g.n for j in 1:g.n if i != j]

function Base.getindex(g :: DenseDiGraph, p)
  i, j = p
  g.adjacency[i, j]
end

getdomain(g :: DenseDiGraph, i) = [false, true]

function Base.setindex!(g :: DenseDiGraph, v, p)
    i, j = p
    g.adjacency[i, j] = v
end

function Base.copy(g :: DenseDiGraph)
  DenseDiGraph(copy(g.adjacency))
end

end
