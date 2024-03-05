import Base.copy, Base.getindex, Base.setindex!, Base.empty
import StatsBase
using SparseArrays
using LinearAlgebra

"""
Directed graphs with a preallocated "scaffold" for calculation of graph statistics.
"""
mutable struct ScaffoldedDirectedGraph{n} <: SampleSpace
    scaffold_edges::SparseMatrixCSC{Bool, Int}
    scaffold_tuples::Set{Tuple{Int, Int}}
    other_edges::Set{Tuple{Int, Int}}

    """
        ScaffoldedDirectedGraph{n}()

    Initialize empty sparse graph with empty scaffold
    """
    function ScaffoldedDirectedGraph{n}() where n
        scaffold_edges = spzeros(Bool, n, n)
        new{n}(spzeros, Set(), Set())
    end

    """
        ScaffoldedDirectedGraph(scaffold)
    
    Use the nonzero elements of matrix `scaffold` to allocate a sparse array for the scaffold.
    """
    function ScaffoldedDirectedGraph(scaffold)
        _scaffold = SparseMatrixCSC(scaffold)
        _scaffold[diagind(_scaffold)] .= false
        dropzeros!(_scaffold)
        r, c = size(_scaffold)
        r == c || error("Scaffold matrix must be square.")

        I, J, V = findnz(_scaffold)
        tuples = Set(zip(I,J))

        new{r}(_scaffold, tuples, Set())
    end
end

function random_index(g::ScaffoldedDirectedGraph{n}) where n
    index = StatsBase.sample(1:n, 2, replace=false)
    tuple(index...)
end

function Base.getindex(g::ScaffoldedDirectedGraph, index)::Bool
    # i,j = index
    # i == j && return false  # no self-loops
    # check the scaffold matrix then check the other edges
    Base.getindex(g.scaffold_edges, index) || index ∈ g.other_edges
end

function Base.setindex!(g::ScaffoldedDirectedGraph, value::Bool, index...)
    length(index) == 2 || error("Can't set index $index")
    i, j = index
    i == j && error("Can't set self-loops in ScaffoldedDirectedGraph")

    if value
        if index ∈ g.scaffold_tuples
            # g.scaffold_edges[index] = value
            setindex!(g.scaffold_edges, value, index...)
        else
            push!(g.other_edges, index)
        end
    else
        if g.scaffold_edges[index] && index ∈ g.scaffold_tuples
            # g.scaffold_edges[index] = value
            setindex!(g.scaffold_edges, value, index...)
        else
            delete!(g.other_edges, index)
        end
    end
    return value
end

# function Base.setindex!(g::ScaffoldedDirectedGraph, value::Bool, index...)
#     length(index) == 2 || error("Can't set index $index")
#     setindex!(g, value, (index[1], index[2]))
# end

function Base.copy(g::ScaffoldedDirectedGraph)
    error("Not implemented")
end