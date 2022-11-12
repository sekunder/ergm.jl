using PyCall
using JLD

function convert_to_jld()
path = "/home/will/research/xaqlab/ergm/G_proof_v6.p"
py"""
import pickle
import networkx as nx
import numpy as np

G = pickle.load(open($path, "rb"))
G = nx.DiGraph(G)
node_ix = {n: i for i, n in enumerate(G.nodes)}
edge_list = np.zeros((G.number_of_edges(), 2), int)
node_positions = np.zeros((G.number_of_nodes(), 3), float)

for ix, edge in enumerate(G.edges()):
   i, j = edge
   edge_list[ix, 0] = node_ix[i]
   edge_list[ix, 1] = node_ix[j]

for i in G.nodes():
   node_positions[node_ix[i], 0] = G.nodes[i]['soma_x']
   node_positions[node_ix[i], 1] = G.nodes[i]['soma_y']
   node_positions[node_ix[i], 2] = G.nodes[i]['soma_z']

"""

edge_list = py"edge_list"
node_positions = py"node_positions"
@save "G_proof_v6.jld" node_positions edge_list
