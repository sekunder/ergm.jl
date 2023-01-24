struct DirectedSpatialTripletModel <: Model
    @doc """
        DirectedSpatialTripletModel(motifs::Vector{Int}, node_embedding::Matrix{Float64}, motif_radius::Float64) 

    Define an ERGM on directed graphs with spatially local triplet motifs as its sufficient statistics.
    
    # Arguments
    - `motifs :: Vec[Int]`: indices of the triplet motifs to include (see Triplet Subgraphs section in docs).
    - `node_embedding :: Matrix{Float64}`: each row node_embedding[i, :] specifies the coordinates of the ith node in a Euclidean space.
    - `motif_radius :: Float64`: only motifs where the Euclidean distance between all pairs of involved nodes is less than motif_radius are
      considered local and are counted in the model statistics.
    """
    function DirectedSpatialTripletModel(motifs::Vector{Int}, node_embedding::Matrix{Float64}, motif_radius::Float64) 
        new()
    end
end

function get_stats(model::DirectedSpatialTripletModel)::Vector{Float64}
    zeros(3)
end