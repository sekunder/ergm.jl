# These tests ensure that the triplet subgraph counts computed by
# a DirectedSpatialTripletModel stay aligned with the triad census
# as computed by the NetworkX Python package.

using ergm.spaces
using ergm.models
using Random
using SparseArrays
using Test
using PyCall
using LinearAlgebra

"Test calculation of sufficient statistics in `set_state`."
function test_set_state()
    n = 100
    G = SparseDirectedGraph(sprand(Bool, n, n, 0.4))
    X = randn((n, 3))
    r = 1.0
    model = DirectedSpatialTripletModel(X, r, zeros(15))
    
    # compute motif counts from ergm.jl
    set_state(model, G)
    actual_motif_counts = model.motif_counts
    
    # convert graph to networkx object
    nx = pyimport("networkx")
    G_nx = nx.DiGraph(Matrix(G.adjacency))

    # convert local graph to networkx object
    GL_nx = nx.DiGraph(Matrix(model.local_state.adjacency))
    
    # compute motif counts from networkx
    edge_count = G_nx.number_of_edges()
    census = nx.triadic_census(GL_nx)
    expected_motif_counts = [
        edge_count,
        round(Int, nx.reciprocity(G_nx) * edge_count) ÷ 2,
        census["021D"], census["021U"], census["021C"],
        census["111D"], census["111U"], census["030T"],
        census["030C"], census["201"], census["120D"],
        census["120U"], census["120C"], census["210"],
        census["300"]
    ]
    
    @test actual_motif_counts == expected_motif_counts
end

"Test incremental updating of sufficient statistics."
function test_updates()
   n = 100
   G = SparseDirectedGraph(sprand(Bool, n, n, 0.4))
   X = randn((n, 3))
   r = 1.0
   model = DirectedSpatialTripletModel(X, r, zeros(15))

   # perform random edge toggles
   for _ ∈ 1:10000
       G = get_state(model)
       index = random_index(G)
       apply_update(model, index, !G[index])
   end

   actual_motif_counts = copy(model.motif_counts)

   # recompute motif counts from scratch
   new_model = DirectedSpatialTripletModel(X, r, zeros(15))
   set_state(new_model, get_state(model))
   expected_motif_counts = new_model.motif_counts

   @test actual_motif_counts == expected_motif_counts
end

@testset "DirectedSpatialTripletModel" begin
    test_set_state()
    test_updates()
end
