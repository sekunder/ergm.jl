module data

using ergm.spaces
using SparseArrays
using PyCall

"""
Save a spatially embedded graph to a file.

The matrix `X` specifies a Euclidean embedding of the graph's nodes.
The ith node of `g` is embedded at the coordinates `X[i, :]`.

The graph is saved as a GraphML file with node positions saved
as node attributes. Note that if you save and then read a graph,
the order of the nodes may not be preserved.
"""
function save_spatial_graph(g::SparseDirectedGraph{n}, X::Matrix{Float64}, filename::String) where n
    edges = [
             (i, j)
             for (i, j, v) ∈ zip(findnz(g.adjacency)...)
             if v != 0
    ]

    py"""
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from([(i + 1, {"pos": list($X[i, :])}) for i in range($n)])
    G.add_edges_from($edges)
    nx.write_gml(G, $filename)
    """
end

"""
Read a spatially embedded graph from a file.

Returns a SparseDirectedGraph `g` specifying connectivity and a
matrix `X` specifying a Euclidean embedding of the graph's nodes.
The ith node of `g` is embedded at the coordinates `X[i, :]`.

The graph must be stored as a GraphML file with node positions saved
as node attributes. For reference, see the output of `save_spatial_graph`,
which is the inverse of this function up to a possible reordering of the nodes.
"""
function load_spatial_graph(filename::String) :: Tuple{SparseDirectedGraph, Matrix{Float64}}
    py"""
    import networkx as nx
    import numpy as np

    G = nx.read_gml($filename)
    A = nx.to_scipy_sparse_matrix(G)
    """

    n = py"""G.number_of_nodes()"""
    is, js = py"""A.nonzero()"""
    is = convert(Vector{Int64}, is) .+ 1
    js = convert(Vector{Int64}, js) .+ 1
    A = sparse(is, js, fill(true, length(is)), n, n)
    G = SparseDirectedGraph(A)
    X = py"""np.vstack([G.nodes[n]["pos"] for n in G.nodes])"""
    G, X
end

end
