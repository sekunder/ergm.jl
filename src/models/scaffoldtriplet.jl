using ergm.spaces
using DataStructures
using SparseArrays
using LinearAlgebra
import Base.show

mutable struct ScaffoldedTripletModel <: Model
    # scaffold::SparseDirectedGraph
    # adj::ScaffoldedDirectedGraph
    parameters::Vector{Float64}

    motif_counts::Vector{Int}   # raw motif counts
    motif_normalizations::Vector{Float64}
    cached_motif_counts::Dict   # possible motif counts if changes are made

    state::ScaffoldedDirectedGraph
    # local_state::SparseDirectedGraph
    # local_state_t::SparseDirectedGraph

    @doc """
        ScaffoldedTripletModel(scaffold::SparseDirectedGraph)
    
    The sufficient statistics of this ERGM are the density of edges, the density of reciprocal edges, and the densities
    of the 13 connected triplet subgraphs in the same order as they are presented in the figure in the documentation
    page of specifying an ERGM. This makes for 15 parameters in total.

    All edges and reciprocal edges are counted when computing the sufficient statistics, but only scaffolded connected
    triplet subgraphs are counted. Such a subgraph is scaffolded if all edges appear in the scaffold graph.
    Note also that the same edge can be included in the edge density and reciprocal
    edge density statistics (as well as involved in connected triplets), but that any set of three nodes can be included
    in at most one of the connected triplet densities.

    An equivalent way to think about this is that the scaffold network acts as a binary mask on the adjacency matrix of
    the current state when computing the statistics.

    All densities are just total counts divided by the number of nodes in the graph, which is the appropriate scaling
    for large spatially local graphs.
    
    # Arguments
    - `scaffold`: The adjacency matrix which counts as "local" for the purposes of counting motifs
    - `parameters`: A vector of 15 real numbers
    - `empty_init`: If true (default), initial state is an empty graph.
    """
    function ScaffoldedTripletModel(scaffold::AbstractMatrix, parameters::Vector{Float64}, empty_init=true)
        length(parameters) == 15 || error("ScaffoldedTripletModel requires exactly 15 parameters")

        _scaffold = ScaffoldedDirectedGraph(scaffold, empty_init)
        if empty_init
            motifs = zeros(Int, 15)
        else
            motifs = triplet_motif_counts(scaffold)
        end

        n = size(scaffold, 1)

        new(parameters,                    # parameters
            motifs,                        # motif counts
            fill(n, 15),                   # motif normalization
            Dict(),                        # cached motif counts
            _scaffold,                     # state
            # SparseDirectedGraph{n}(),      # local state
            # SparseDirectedGraph{n}(),      # local state transpose
            )
    end
end

function Base.show(io::IO, M::ScaffoldedTripletModel)
    r,c = size(M.state.scaffold_edges)
    print(io, "ScaffoldedERGM on $r nodes with $(length(M.parameters)) parameters")
end

@doc """
    set_normalization(m::ScaffoldedTripletModel, z::Real)
    set_normalization(m::ScaffoldedTripletModel, v::Vector)

Set the normalizations for the model; either all to the same value z or to the specified vector
"""
function set_normalization(m::ScaffoldedTripletModel, z::Real)
    m.motif_normalizations = fill(Float64(z), 15)
end
function set_normalization(m::ScaffoldedTripletModel, v::AbstractVector)
    length(v) == 15 || error("Normalizations must be length 15")
    m.motif_normalizations = v
end

function get_sample_space(model::ScaffoldedTripletModel)
    # n = size(model.state)
    # SparseDirectedGraph{n}
    typeof(model.state)
end

function get_state(model::ScaffoldedTripletModel)
    model.state
end

function get_parameters(model::ScaffoldedTripletModel)
    model.parameters
end

function set_parameters(model::ScaffoldedTripletModel, parameters::Vector{Float64})
    model.parameters = parameters
end

function get_statistics(model::ScaffoldedTripletModel)
    model.motif_counts ./ model.motif_normalizations
end

function set_state(model::ScaffoldedTripletModel, state::ScaffoldedDirectedGraph)
    model.state = copy(state)

    # what needs to be happen:
    # 1. TODO Compute the stats of the masked adjacency matrix and store them
    model.motif_counts = triplet_motif_counts(model.state.scaffold_edges)
    # for other_edge in model.state.other_edges
    #     scaff_stats[1] += 1
    # end
    model.motif_counts[1] += length(model.state.other_edges)
    for edge in model.state.other_edges
        # count reciprocal pairs where at least one edge is outside the scaffold.
        model.motif_counts[1] += model.state[edge[2], edge[1]]
    end
    
    # 2. Invalidate cache
    #TODO This would seem to allocate a new object every time this function is called. Can this be done with an in-place clear?
    model.cached_motif_counts = Dict()

    return nothing
end

function test_update(model::ScaffoldedTripletModel, index::Tuple{Int, Int}, value::Bool; normalized=true)
    if (index, value) ∈ keys(model.cached_motif_counts)
        return model.cached_motif_counts[(index, value)]
    end

    new_stats = copy(model.motif_counts)

    # update the edge count and reciprocal edge count
    new_stats[1] += value - model.state[index]
    if model.state[reverse(index)]
        new_stats[2] += value - model.state[index]
    end

    if index ∈ model.state.scaffold_tuples
        if model.state[index] != value
            # we are trying to toggle an edge that affects the motif counts
            Δstats = delta_triplet_motif_counts(model.state.scaffold_edges, index)
        else
            # the proposed change is not actually a change
            Δstats = zeros(Int, 15)
        end
        # update the counts of triplets in the scaffold subgraph
        new_stats[3:end] = new_stats[3:end] + Δstats[3:end]
    end

    model.cached_motif_counts[(index, value)] = new_stats

    if normalized
        return new_stats ./ model.motif_normalizations
    else
        return new_stats
    end
end

function apply_update(model::ScaffoldedTripletModel, index::Tuple{Int, Int}, value::Bool)
    # compute or fetch new motif counts
    model.motif_counts = test_update(model, index, value; normalized=false)

    # update the state graph
    model.state[index] = value

    # invalidate cache
    model.cached_motif_counts = Dict()
    return nothing
end
