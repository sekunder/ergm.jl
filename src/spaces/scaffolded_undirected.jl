import Base.copy, Base.getindex, Base.setindex!, Base.empty, Base.show
import StatsBase
using SparseArrays
using LinearAlgebra

mutable struct ScaffoldedUndirectedGraph{N} <: SampleSpace
    scaffold_edges::SparseMatrixCSC{Bool, Int}
    scaffold_tuples::Set{Tuple{Int, Int}}
    other_edges::Set{Tuple{Int, Int}}

    """
        ScaffoldedUndirectedGraph{n}()

    Initialize empty sparse graph with empty scaffold
    """
    function ScaffoldedUndirectedGraph{N}() where N
        _scaffold_edges = spzeros(Bool, N, N)
        new{N}(_scaffold_edges, Set{Tuple{Int, Int}}(), Set{Tuple{Int, Int}}())
    end

    """
        ScaffoldedUndirectedGraph(scaffold)
    
    Use the nonzero off-diagonal elements of matrix `scaffold` to allocate a sparse array for the scaffold.
    If `empty_graph=true`, initialize with an empty graph
    """
    function ScaffoldedUndirectedGraph(scaffold::AbstractMatrix,
                                       empty_graph=false,
                                       extra_edges=Set{Tuple{Int,Int}}())
        R, C = size(scaffold)
        R == C || error("Scaffold matrix must be square.")
        _scaffold = SparseMatrixCSC(scaffold)
        _scaffold[diagind(_scaffold)] .= false
        dropzeros!(_scaffold)

        I, J, V = findnz(_scaffold)
        tuples = Set((min(i, j), max(i, j)) for (i, j) in zip(I, J))

        if empty_graph
            _scaffold[:] .= 0
            extra_edges = Set{Tuple{Int,Int}}()
        end

        new{R}(_scaffold, tuples, extra_edges)
    end
end

function Base.show(io::IO, G::ScaffoldedUndirectedGraph{N}) where N
    m_prealloc = length(G.scaffold_tuples)
    m_other = length(G.other_edges)
    m_scaff = sum(G.scaffold_edges) ÷ 2  # Each edge is counted twice in the adjacency matrix
    print(io, "ScaffoldedUndirectedGraph($N nodes, $(m_scaff + m_other) edges ($m_scaff / $m_prealloc preallocated, $m_other other))")
end

function random_index(::ScaffoldedUndirectedGraph{N}) where N
    index = StatsBase.sample(1:N, 2, replace=false)
    (min(index...), max(index...))
end

function Base.getindex(g::ScaffoldedUndirectedGraph, index::Tuple{Int, Int})::Bool
    index = (min(index...), max(index...))
    Base.getindex(g.scaffold_edges, index...) || index ∈ g.other_edges
end

function Base.setindex!(g::ScaffoldedUndirectedGraph, value::Bool, index::Tuple{Int, Int})
    index = (min(index...), max(index...))
    length(index) == 2 && index[1] == index[2] && error("Can't set self-loops in ScaffoldedUndirectedGraph")

    if value
        if index ∈ g.scaffold_tuples
            g.scaffold_edges[index...] = value
            g.scaffold_edges[reverse(index)...] = value
            # setindex!(g.scaffold_edges, value, index...)
            # setindex!(g.scaffold_edges, value, reverse(index)...)
        else
            push!(g.other_edges, index)
        end
    else
        if g.scaffold_edges[index] && index ∈ g.scaffold_tuples
            g.scaffold_edges[index...] = value
            g.scaffold_edges[reverse(index)...] = value
            # setindex!(g.scaffold_edges, value, index...)
            # setindex!(g.scaffold_edges, value, reverse(index)...)
        else
            delete!(g.other_edges, index)
        end
    end
end

function Base.copy(g::ScaffoldedUndirectedGraph)
    ScaffoldedUndirectedGraph(copy(g.scaffold_edges), false, copy(g.other_edges))
end




