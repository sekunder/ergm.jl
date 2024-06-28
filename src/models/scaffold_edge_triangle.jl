using ergm.spaces
using DataStructures
using SparseArrays
using LinearAlgebra
import Base.show

mutable struct ScaffoldedEdgeTriangleModel <: Model
    parameters:: Vector{Float64}
    motif_counts::Vector{Int}   # counts of edges and triangles
    motif_normalizations::Vector{Float64}
    cached_motif_counts::Dict   # possible motif counts if changes are made

    state::ScaffoldedUndirectedGraph

    function ScaffoldedEdgeTriangleModel(scaffold::AbstractMatrix, parameters::Vector{Float64}, empty_init=true)
        length(parameters) == 2 || error("ScaffoldedEdgeTriangleModel requires exactly 2 parameters")
        
        _scaffold = ScaffoldedUndirectedGraph(scaffold, empty_init)
        if empty_init
            motifs = zeros(Int, 2)
        else
            motifs = [sum(scaffold), tr(scaffold ^ 3)]
        end

        n = size(scaffold, 1)

        new(parameters, motifs, [1, n], Dict(), _scaffold)
    end
end

function Base.show(io::IO, M::ScaffoldedEdgeTriangleModel)
    r,c = size(M.state.scaffold_edges)
    print(io, "ScaffoldedEdgeTriangleModel on $r nodes and 2 parameters")
end

function get_sample_space(m::ScaffoldedEdgeTriangleModel)
    typeof(m.state)
end

function get_parameters(m::ScaffoldedEdgeTriangleModel)
    m.parameters
end

function set_parameters(m::ScaffoldedEdgeTriangleModel, parameters::Vector{Float64})
    m.parameters = parameters
end

function get_statistics(m::ScaffoldedEdgeTriangleModel)
    m.motif_counts ./ m.motif_normalizations
end

function get_state(m::ScaffoldedEdgeTriangleModel)
    m.state
end

function set_state(m::ScaffoldedEdgeTriangleModel, state::ScaffoldedUndirectedGraph{N} where N)
    m.state = state

    m.motif_counts = [sum(state.scaffold_edges) + length(state.other_edges), tr(state.scaffold_edges^3)]

    m.cached_motif_counts = Dict()
end

function test_update(m::ScaffoldedEdgeTriangleModel, index, value; normalized=true)
    if (index, value) ∈ keys(m.cached_motif_counts)
        return m.cached_motif_counts[(index, value)]
    end

    new_stats = copy(m.motif_counts)

    Delta_uv = value - m.state[index]

    new_stats[1] += Delta_uv

    if index ∈ m.state.scaffold_tuples
        new_stats[2] += Delta_uv * (m.state.scaffold_edges^2)[index]
    end
    
    m.cached_motif_counts[(index, value)] = new_stats

    if normalized
        return new_stats ./ m.motif_normalizations
    else
        return new_stats
    end
end

function apply_update(m::ScaffoldedEdgeTriangleModel, index, value)
    m.motif_counts = test_update(m, index, value, normalized=false)
    m.state[index] = value
    m.cached_motif_counts = Dict()
    return nothing
end