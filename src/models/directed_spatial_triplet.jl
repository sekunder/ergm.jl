using ergm.spaces
using DataStructures
using SparseArrays
using LinearAlgebra

mutable struct DirectedSpatialTripletModel <: Model
    node_embedding::Matrix{Float64}
    motif_radius::Float64
    parameters::Vector{Float64}

    motif_counts::Vector{Int}
    motif_normalizations::Vector{Float64}
    cached_motif_counts::Dict

    state::SparseDirectedGraph
    local_state::SparseDirectedGraph
    local_state_t::SparseDirectedGraph
    
    @doc """
        DirectedSpatialTripletModel(node_embedding::Matrix{Float64}, motif_radius::Float64) 

    The sufficient statistics of this ERGM are the density of edges, the density of reciprocal edges, and the densities
    of the 13 connected triplet subgraphs in the same order as they are presented in the figure in the documentation
    page of specifying an ERGM. This makes for 15 parameters in total.

    All edges and reciprocal edges are counted when computing the sufficient statistics, but only _local_ connected
    triplet subgraphs are counted. Such a subgraph is considered local if the distance between all pairs of involved
    nodes is less than `motif_radius`. Note also that the same edge can be included in the edge density and reciprocal
    edge density statistics (as well as involved in connected triplets), but that any set of three nodes can be included
    in at most one of the connected triplet densities.

    All densities are just total counts divided by the number of nodes in the graph, which is the appropriate scaling
    for large spatially local graphs.
    
    # Arguments
    - `node_embedding::Matrix{Float64}`: each row node_embedding[i, :] specifies the coordinates of the ith node in a Euclidean space.
    - `motif_radius::Float64`: defines what connected triplet subgraphs are considered local, as described above.
    - `parameters::Vector{Float64}`: natural paramters corresponding to each of the 15 sufficient statistics.
    """
    function DirectedSpatialTripletModel(node_embedding::Matrix{Float64}, motif_radius::Float64, parameters::Vector{Float64}) 
        if length(parameters) != 15
            error("DirectedSpatialTripletModel requires 15 natural parameters.")
        end

        n = size(node_embedding, 1)

        new(
            node_embedding, motif_radius, parameters,
            zeros(Int, 15), fill(n, 15), Dict(),
            SparseDirectedGraph{n}(), SparseDirectedGraph{n}(), SparseDirectedGraph{n}()
        )
    end
end

function get_sample_space(model::DirectedSpatialTripletModel)
    number_of_nodes = size(model.node_embedding, 1)
    SparseDirectedGraph{number_of_nodes}
end

function get_state(model::DirectedSpatialTripletModel)::SparseDirectedGraph
    model.state
end

_triplet_codes = [
     1,  2,  2,  3,  2,  4,  6,  8,
     2,  6,  5,  7,  3,  8,  7, 11,
     2,  6,  4,  8,  5,  9,  9, 13,
     6, 10,  9, 14,  7, 14, 12, 15,
     2,  5,  6,  7,  6,  9, 10, 14,
     4,  9,  9, 12,  8, 13, 14, 15,
     3,  7,  8, 11,  7, 12, 14, 15,
     8, 14, 13, 15, 11, 15, 15, 16
]

"""Convert 6-bit subgraph code to index of corresponding statistic.

Any three nodes {u, v, w} give rise to a 6-bit code, where
each bit wu|uw|wv|vw|uv|vu indicates whether the corresponding
directed edge is present in our graph. These 64 possible 6-bit
codes map onto the 16 isomorphism classes of three-node graphs.

Note that this function only returns the indices of the sufficient
statistics corresponding to connected local triplet subgraphs. If the
provided 6-bit code corresponds to a non-connected subgraph, this
function returns `nothing`.
"""
function decode_subgraph(code::UInt8)::Union{Nothing, Int}
    triplet_type = _triplet_codes[code + 1] 

    # return nothing if code corresponds to a non-connected subgraph
    if triplet_type <= 3
        return nothing
    end

    # shift so that subgraph_type of 4 (the first connected triplet subgraph)
    # corresponds to the 3rd sufficient statistic
    triplet_type - 1
end

"Compute spatial distance between nodes."
function distance(model::DirectedSpatialTripletModel, i::Int, j::Int)::Float64
    norm(model.node_embedding[i, :] - model.node_embedding[j, :])
end

"""
Compute the triplet codes of local triplet subgraphs involving a given edge.

Let `index = (u, v)`. If `avoid_recount` is set to `true`, then only count triplet subgraphs where the 
third node `w` satisfies `w > max(u, v)`. This is useful to avoid double-counting any triplet subgraphs
when computing the original triplet subgraph counts in `set_state`.

Return a dictionary mapping index `w` of the third node to the subgraph code for the triplet `(u, v, w)`.
"""
function triplet_codes_involving_edge(model::DirectedSpatialTripletModel, index::Tuple{Int, Int}; avoid_recount::Bool=false)::Dict{Int, UInt8}
    LA = model.local_state.adjacency
    LA_t = model.local_state_t.adjacency
    u, v = index
    uv = UInt8(LA[u, v])
    vu = UInt8(LA[v, u])

    # part of 6-bit subgraph code depending on uv and vu
    base_code = UInt8(0)
    base_code |= vu << 0
    base_code |= uv << 1

    triplet_codes = DefaultDict{Int, UInt8}(base_code)

    # set vw bits
    for w ∈ findall(LA_t[:, v])
        # always ensure w not equal to u (w will never equal v because graph has
        # no self-loops), and ensure w > max(u, v) if avoid_recount is true
        if w == u || avoid_recount && w <= max(u, v)
            continue
        end

        triplet_codes[w] |= UInt8(1) << 2
    end

    # set wv bits
    for w ∈ findall(LA[:, v])
        # always ensure w not equal to u (w will never equal v because graph has
        # no self-loops), and ensure w > max(u, v) if avoid_recount is true
        if w == u || avoid_recount && w <= max(u, v)
            continue
        end

        triplet_codes[w] |= UInt8(1) << 3
    end

    # set uw bits
    for w ∈ findall(LA_t[:, u])
        # always ensure w not equal to v (w will never equal u because graph has
        # no self-loops), and ensure w > max(u, v) if avoid_recount is true
        if w == v || avoid_recount && w <= max(u, v)
            continue
        end

        triplet_codes[w] |= UInt8(1) << 4
    end

    # set wu bits
    for w ∈ findall(LA[:, u])
        # always ensure w not equal to v (w will never equal u because graph has
        # no self-loops), and ensure w > max(u, v) if avoid_recount is true
        if w == v || avoid_recount && w <= max(u, v)
            continue
        end

        triplet_codes[w] |= UInt8(1) << 5
    end

    triplet_codes
end

function set_state(model::DirectedSpatialTripletModel, state::SparseDirectedGraph)
    model.state = copy(state)

    # compute local version of state graph that
    # only includes edges with length shorter
    # than model.motif_radius
    A = state.adjacency
    i, j, v = findnz(A)

    # filter out non-local edges
    lix = [
        ix for ix ∈ 1:length(i) 
        if distance(model, i[ix], j[ix]) < model.motif_radius
    ]
    LA = sparse(i[lix], j[lix], v[lix], size(A)...)
    model.local_state = SparseDirectedGraph(LA)

    # also store transposed version of local graph
    LA_t = SparseMatrixCSC{Bool, Int}(LA')
    model.local_state_t = SparseDirectedGraph(LA_t)

    # reset motif counts
    model.motif_counts = zeros(15)

    # compute edge count
    model.motif_counts[1] = sum(A)

    # compute reciprocal edge count
    model.motif_counts[2] = sum(A .* A') ÷ 2

    # compute triplet subgraph counts
    n = size(model.node_embedding, 1)

    for u ∈ 1:n
        for v ∈ u+1:n
            triplet_codes = triplet_codes_involving_edge(model, (u, v); avoid_recount=true)
            for code ∈ values(triplet_codes)
                statistic_index = decode_subgraph(code)

                # ignore non-connected triplet subgraphs
                if !isnothing(statistic_index)
                    model.motif_counts[decode_subgraph(code)] += 1
                end
            end
        end
    end

    # invalidate cache
    model.cached_motif_counts = Dict()
    return nothing
end

function get_statistics(model::DirectedSpatialTripletModel)::Vector{Float64}
    model.motif_counts ./ model.motif_normalizations
end

function test_update(model::DirectedSpatialTripletModel, index::Tuple{Int, Int}, value::Bool; counts=false)
    if (index, value) ∈ keys(model.cached_motif_counts)
        return model.cached_motif_counts[(index, value)]
    end

    new_counts = copy(model.motif_counts)
    A = model.state.adjacency
    u, v = index
    uv = A[u, v]
    vu = A[v, u]

    # change in edge count
    new_counts[1] += value - uv

    # change in reciprocal edge count
    if vu
        new_counts[2] += value - uv
    end

    # compute updated connected triplet statistics only
    # if the edge uv is considered local
    if distance(model, u, v) < model.motif_radius
        triplet_codes = triplet_codes_involving_edge(model, index)

        for code ∈ values(triplet_codes)
            # updated_code is code with uv bit set
            # to the updated value of the edge uv
            updated_code = code & ~(UInt8(1) << 1)
            updated_code |= UInt8(value) << 1

            statistic_index = decode_subgraph(code)

            # ignore non-connected triplet subgraphs
            if !isnothing(statistic_index)
                new_counts[statistic_index] -= 1
            end

            updated_statistic_index = decode_subgraph(updated_code)

            # ignore non-connected triplet subgraphs
            if !isnothing(updated_statistic_index)
                new_counts[updated_statistic_index] += 1
            end
        end
    end

    model.cached_motif_counts[(index, value)] = new_counts

    # if counts is true, return un-normalized counts
    if counts
        new_counts
    else
        new_counts ./ model.motif_normalizations
    end
end

function apply_update(model::DirectedSpatialTripletModel, index::Tuple{Int, Int}, value::Bool)
    # compute new motif counts
    model.motif_counts = test_update(model, index, value; counts=true)
    u, v = index

    # update the three state graphs
    model.state[(u, v)] = value
    
    if distance(model, u, v) < model.motif_radius
        model.local_state[(u, v)] = value
        model.local_state_t[(v, u)] = value
    end

    # invalidate cache
    model.cached_motif_counts = Dict()
    return nothing
end

function get_parameters(model::DirectedSpatialTripletModel)::Vector{Float64}
    model.parameters
end

function set_parameters(model::DirectedSpatialTripletModel, parameters::Vector{Float64})
    model.parameters = parameters
end
