module data

using ergm.spaces
using SparseArrays
using PyCall

export example_graph, example_graph_names

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


"""
    read_edgelist_file(filename; directed=false)

Reads an edgelist file into a sparse, Boolean adjacency matrix.

Each edge is a pair of integers separated by whitespace. Each line contains one edge.
The largest integer that appears is treated as the number of nodes.

First, it reads the entire file into a matrix. Then, uses the columns of that matrix to instantiate a sparse matrix using `sparse`.
If `directed=false`, the data is used as-is and the resulting sparse matrix is returned.
If `directed=true`, the data is duplicated with all edges reversed, so each edge is represented in the "forward" and "backward" direction.
"""
function read_edgelist_file(filename; directed=false)
    n_lines = countlines(filename)
    data = zeros(Int, n_lines, 2)
    for (line_no, line) in enumerate(eachline(filename))
        tks = split(line)
        data[line_no, 1] = parse(Int, tks[1])
        data[line_no, 2] = parse(Int, tks[2])
    end
    n = maximum(data)
    if directed
        return sparse(data[:, 1], data[:, 2], trues(n_lines), n, n)
    else
        data = cat(data, data[:,[2,1]], dims=1)
        return sparse(data[:, 1], data[:, 2], trues(2 * n_lines), n, n)
    end
end


const ergm_example_files = Dict(
    "karate" => "karate.txt",
    "larvalMB" => "larval_MB_undirected.txt",
    "larvalMB0.0" => "larval_MB_undirected_scaffold_0.0.txt",
    "larvalMB0.05" => "larval_MB_undirected_scaffold_0.05.txt",
    "larvalMB0.1" => "larval_MB_undirected_scaffold_0.1.txt",
    "larvalMB0.25" => "larval_MB_undirected_scaffold_0.25.txt",
    "larvalMB0.5" => "larval_MB_undirected_scaffold_0.25.txt",
    "larvalMB0.75" => "larval_MB_undirected_scaffold_0.75.txt",
    "larvalMB1.0" => "larval_MB_undirected_scaffold_1.0.txt")

"""
    example_graph(name)

The adjacency matrix of the named sample graph. By default `name="karate"`

Options:
 - `"karate"`: Zachary's Karate Club network
 - `"larvalMB"`: Larval mushroom body network
 - `"larvalMB*"` where `*` can be one of `0.0`, `0.05`, `0.1`, `0.25`, `0.5`, `0.75`: The larval mushroom body network, only including those edges which connect nodes in the same cluster.

 Full list of example graph names can be accessed with `example_graph_names()`
"""
function example_graph(name="karate")
    filename = get(ergm_example_files, name, "Invalid name: $name")
    return read_edgelist_file("src/data/$filename")
end

example_graph_names() = keys(ergm_example_files)

end