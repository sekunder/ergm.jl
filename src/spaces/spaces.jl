module spaces

import Base

# define general interface for spaces
abstract type SampleSpace end

"""
    random_index(g::SampleSpace)

Sample the index of an edge uniformly. The edge need not actually be present.
"""
function random_index(g::SampleSpace)
    error("unimplemented")
end

"""
    getindex(g::SampleSpace, index)

Retrieve value of edge with given index.
"""
function Base.getindex(g::SampleSpace, index)
    error("unimplemented")
end

"""
    setindex!(g::SampleSpace, value, index)

Set value of edge with given index.
"""
function Base.setindex!(g::SampleSpace, value, index)
    error("unimplemented")
end

"""
    copy(g::SampleSpace)

Perform a deep copy of an element of the space.
"""
function Base.copy(g::SampleSpace)
    SparseGraph(copy(g.adjacency))
end


"""
    adjacency_matrix(g::SampleSpace)

The adjacency matrix of the current state of `g`.
"""
function adjacency_matrix(g::SampleSpace)
    error("adjacency_matrix not implemented for type $(typeof(g))")
end

export SampleSpace, random_index, adjacency_matrix
    
# export pre-defined models

include("sparse_directed.jl")
export SparseDirectedGraph
include("scaffolded_directed.jl")
export ScaffoldedDirectedGraph

include("scaffolded_undirected.jl")
export ScaffoldedUndirectedGraph

end