module platinum

export load_graph, subsample, nx_to_digraph, nx_to_ps

using PyCall
using ergm.spaces

function load_graph()
    py"""
    import pickle
    import networkx as nx
    G = pickle.load(open("/home/will/research/xaqlab/ergm/G_proof_v6.p", "rb"))
    """
    py"G"
end

function subsample(G, k, s)
    py"""
    import numpy as np
    import numpy.linalg as npl
    import numpy.random as npr

    n = $G.number_of_nodes()
    ps = np.vstack([
        np.array([G.nodes[n]['soma_' + i] for i in 'xyz'])
        for n in G.nodes
    ])
    all_ns = list(G.nodes)
    Hs = []
        
    for _ in range($s):
        j0 = npr.randint(n)
        ds = npl.norm(ps - ps[j0, :], axis=1)
        js = np.argpartition(ds, $k)[:$k]
        ns = [all_ns[j] for j in js]
        Hs.append(G.subgraph(ns))
    """
    py"Hs"
end

function nx_to_digraph(nx)
    A = py"nx.adjacency_matrix($nx).todense()"
    DiGraph(A .> 0)
end

function nx_to_ps(nx)
    py"""
    ps = np.vstack([
        np.array([G.nodes[n]['soma_' + i] for i in 'xyz'])
        for n in $nx.nodes
    ])
    """
    py"ps"
end

end
