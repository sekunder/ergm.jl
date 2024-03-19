import Base.copy, Base.getindex, Base.setindex!, Base.empty, Base.show
import StatsBase
using SparseArrays
using LinearAlgebra

"""
Directed graphs with a preallocated "scaffold" for calculation of graph statistics.
"""
mutable struct ScaffoldedDirectedGraph{N} <: SampleSpace
    scaffold_edges::SparseMatrixCSC{Bool, Int}
    scaffold_tuples::Set{Tuple{Int, Int}}
    other_edges::Set{Tuple{Int, Int}}

    """
        ScaffoldedDirectedGraph{n}()

    Initialize empty sparse graph with empty scaffold
    """
    function ScaffoldedDirectedGraph{N}() where N
        _scaffold_edges = spzeros(Bool, N, N)
        new{N}(_scaffold_edges, Set(), Set(), Set())
    end

    """
        ScaffoldedDirectedGraph(scaffold)
    
    Use the nonzero off-diagonal elements of matrix `scaffold` to allocate a sparse array for the scaffold.
    If `empty_graph=true`, initialize with an empty graph
    """
    function ScaffoldedDirectedGraph(scaffold::AbstractMatrix,
                                     empty_graph=false,
                                     extra_edges=Set{Tuple{Int,Int}}())
        R, C = size(scaffold)
        R == C || error("Scaffold matrix must be square.")
        _scaffold = SparseMatrixCSC(scaffold)
        _scaffold[diagind(_scaffold)] .= false
        dropzeros!(_scaffold)

        I, J, V = findnz(_scaffold)
        tuples = Set(zip(I,J))

        if empty_graph
            _scaffold[:] .= 0
            extra_edges = Set{Tuple{Int,Int}}()
        end

        new{R}(_scaffold, tuples, extra_edges)
    end
end

function Base.show(io::IO, G::ScaffoldedDirectedGraph{N}) where N
    m_scaff = length(G.scaffold_tuples)
    m_other = length(G.other_edges)
    print(io, "ScaffoldedDirectedGraph($N nodes, $(m_scaff+m_other) edges ($m_scaff preallocated + $m_other other))")
end

function random_index(::ScaffoldedDirectedGraph{N}) where N
    index = StatsBase.sample(1:N, 2, replace=false)
    tuple(index...)
end

function Base.getindex(g::ScaffoldedDirectedGraph, index::Tuple{Int, Int})::Bool
    # i,j = index
    # i == j && return false  # no self-loops
    # check the scaffold matrix then check the other edges
    Base.getindex(g.scaffold_edges, index...) || index ∈ g.other_edges
end

function Base.setindex!(g::ScaffoldedDirectedGraph, value::Bool, index::Tuple{Int, Int})
    # length(index) <= 2 || error("Can't set index $index")
    # i, j = index
    length(index) == 2 && index[1] == index[2] && error("Can't set self-loops in ScaffoldedDirectedGraph")

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
end

function Base.copy(g::ScaffoldedDirectedGraph)
    # error("Base.copy is incorrectly implemented for ScaffoldedDirectedGraph")
    ScaffoldedDirectedGraph(copy(g.scaffold_edges), false, copy(g.other_edges))
end