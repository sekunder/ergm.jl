using ergm.spaces
using DataStructures
using SparseArrays
using LinearAlgebra

# entry i,j counts the number of subgraphs of motif j that are isomorphic to motif i
#                                               0 1 2 3 4 5 6
#                             1 2 3 4 5 6 7 8 9 1 1 1 1 1 1 1
const exact_to_over_3node  = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1; # 1  003  empty graph
                              0 1 2 2 2 2 3 3 3 3 4 4 4 4 5 6; # 2  012  single edge
                              0 0 1 0 0 0 1 1 0 0 2 1 1 1 2 3; # 3  102  single reciprocal
                              0 0 0 1 0 0 0 1 1 0 1 1 2 1 2 3; # 4  021D diverging
                              0 0 0 0 1 0 1 0 1 0 1 2 1 1 2 3; # 5  021U converging
                              0 0 0 0 0 1 1 1 1 3 2 2 2 3 4 6; # 6  021C two step path
                              0 0 0 0 0 0 1 0 0 0 2 2 0 1 3 6; # 7  111D single edge into a reciprocal
                              0 0 0 0 0 0 0 1 0 0 2 0 2 1 3 6; # 8  111U single edge ouf of a reciprocal
                              0 0 0 0 0 0 0 0 1 0 0 2 2 1 3 6; # 9  030T directed clique, diverging to a single edge
                              0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 2; # 10 030C directed cycle
                              0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 3; # 11 201  reciprocal path
                              0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 3; # 12 120D diverging to a reciprocal
                              0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 3; # 13 120U converging from a reciprocal
                              0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 6; # 14 120C path across a reciprocal
                              0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 6; # 15 210  all but one edge
                              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1] # 16 300  all six edges
const over_to_exact_3node = Matrix{Int}(inv(exact_to_over_3node))

export exact_to_over_3node, over_to_exact_3node

"""
    triplet_motif_counts(A::AbstractMatrix, isomorphism=True, include_empty=False)

Compute the number of occurrences of each isomorphism class of nonempty three-node graphs in A.
Assumes that `A[i,j]` indicates the presence of edge `i -> j`. Will return incorrect results if `A`
has any nonzero diagonal entries.

By default, counts subgraph _isomorphisms_, that is, the number of _induced_ copies of each motif.
To turn off this behavior, pass `isomorphism=false`
Returns a vector of length 15, which does not count the empty triad.

# Arguments
- `A` adjacency matrix for directed graph. `A[i,j]` indicates the present of edge `i -> j`
- `isomorphism` Whether to count subgraph isomorphisms (`True`) or subgraph monomorphisms (`False`)

# Isomorphism vs. Monomorphism
A subgraph H of a graph G is a subset of the vertices and edges of G which is still a graph.
Counting monomorphisms counts subgraphs. For example, every digraph has `n * (n - 1) / 2` copies
of the empty graph E_2 as subgraphs.
Counting induced subgraphs amounts to applying a fancy variation of the principle of inclusion-exclusion
to the monomorphism counts; monomorphism counts are easy to obtain with matrix algebra.
"""
function triplet_motif_counts(A::AbstractMatrix; isomorphism=true, include_empty=false)
    n1, n2 = size(A)
    n1 == n2 || error("Matrix must be square")
    # any(A[diagind(A)] .!= 0) || error("Matrix must have nonzero diagonal")
    n = n1
    m = sum(A)  # number of edges
    # div[i,j] = # k such that k->i and k->j for i != j. div[i,i] = in-degree of i.
    div = A' * A  # common inputs.
    # conv[i,j] = #k such that i->k and j->k for i !=j. conv[i,i] = out-degree of i.
    conv = A * A'  # common outputs
    A2 = A ^ 2
    trA2 = tr(A2)
    A3 = A ^ 3
    R = A .* A'  # reciprocal connections only
    # D = sum(R; dims=2)  # bidirectional degree of each vertex
    R2 = R ^ 2

    counts = zeros(Int, 16)

    counts[1] = n * (n - 1) * (n - 2) ÷ 6  # empty graph; gets truncated away when returned

    counts[2] = (n - 2) * m  # single edge + disconnected third node
    counts[3] = (n - 2) * trA2 ÷ 2  # single reciprocal connection + disconnected third node

    counts[4] = (sum(div) - m) ÷ 2  # diverging
    counts[5] = (sum(conv) - m) ÷ 2  # converging
    counts[6] = sum(A2) - trA2  # sum off-diagonal elements of A^2, counts two-step paths
    # counts[7] = sum(A .* D') - sum(R)  # reciprocal + incoming edge
    # counts[8] = sum(A .* D) - sum(R)  # reciprocal + outgoing edge
    # counts[7] = sum(A * R) - sum(D) ÷ 2
    # counts[8] = sum(R * A) - sum(D) ÷ 2
    counts[7] = sum(A * R) - tr(A * R)
    counts[8] = sum(R * A) - tr(R * A)
    counts[9] = sum(A .* A2)  # directed clique; a -> b, a -> c, b-> c

    counts[10] = tr(A3) ÷ 3  # directed cycle
    
    counts[11] = (sum(R2) - tr(R2)) ÷ 2  # reciprocal path
    counts[12] = sum(R .* div) ÷ 2  # diverging to a reciprocal
    counts[13] = sum(R .* conv) ÷ 2  # converging from a reciprocal
    counts[14] = sum(R .* A2)  # path + recriprocal

    
    counts[15] = sum(R2 .* A)  # reciprocal path + edge
    counts[16] = tr(R2 * R) ÷ 6  # fully connected 

    if include_empty
        counts = isomorphism ? over_to_exact_3node * counts : counts
    else
        counts = isomorphism ? over_to_exact_3node[2:16, 2:16] * counts[2:16] : counts[2:16]
    end
end


"""
    delta_triplet_motif_counts(A::AbstractMatrix, index; isomorphism=true, include_empty=false)

Compute change in motif counts if edge `index` is toggled.
Specifically, if `B = A` except that `B[u,v] = !A[u,v]`,
then this returns motif counts in `B` minus motif counts in `A`.
"""
function delta_triplet_motif_counts(A::AbstractMatrix, index; isomorphism=true, include_empty=false)
    n1, n2 = size(A)
    n1 == n2 || error("Matrix must be square")
    n = n1
    # m = sum(A)
    u, v = index
    Δuv = 1 - 2 * A[u,v]
    common_post = A[u, :] .* A[v, :] # bitvector indicating common out-neighbors of u,v
    common_pre = A[:, u] .* A[:, v]  # bitvector indicating common in-neighbors of u,v
    bideg_u = A[u, :]' * A[:, u] # number of reciprocal partners of u
    bideg_v = A[v, :]' * A[:, v] # number of reciprocal partners of v
    out_u = sum(A[u, :]) # number of out-neighbors of u
    in_u = sum(A[:, u])  # number of in-neighbors of u
    out_v = sum(A[v, :]) # number of out-neighbors of v
    in_v = sum(A[:, v])  # number if in-neighbors of v

    # delta is change in subgraph monomorphisms
    # It counts how many subgraphs of A include ordered pair uv and are isomorphic to each motif.
    delta = zeros(Int, 16)

    # delta[1] = 0  # empty graph monomorphism count does not change

    # single connected pairs
    delta[2] = (n - 2)  # single edge
    delta[3] = (n - 2) * A[v, u]  # single reciprocal

    # two edges and all nodes connected
    delta[4] = out_u - A[u, v]  # two edges diverging from a single source
    delta[5] = in_v - A[u, v]  # two edges converging to a single sink
    delta[6] = in_u + out_v - 2 * A[v,u]  # two step path

    # delta[7] = bideg_v + A[v, u] * (in_u + in_v - 2 * A[u, v] - 2 * A[v, u] + 1)
    delta[7] = bideg_v + A[v, u] * (in_u + in_v - 2 * A[u, v] - 1)
    # delta[8] = bideg_u + A[v, u] * (out_u + out_v - 2 * A[u, v] - 2 * A[v, u] + 1)
    delta[8] = bideg_u + A[v, u] * (out_u + out_v - 2 * A[u, v] - 1)
    delta[9] = sum(common_post) + sum(common_pre) + A[u, :]' * A[:, v]
    delta[10] = A[v, :]' * A[:, u]
    delta[11] = A[v, u] * (bideg_u + bideg_v - 2 * A[u, v])
    delta[12] = common_post' * A[:, v] + A[v, u] * sum(common_pre)
    delta[13] = A[u, :]' * common_pre + A[v, u] * sum(common_post)
    delta[14] = common_post' * A[:, u] + A[v, :]' * common_pre
    delta[15] = common_post' * common_pre

    if A[v, u] != 0
        delta[14] = delta[14] + A[u, :]' * A[:, v] + A[v, :]' * A[:, u]
        delta[15] = delta[15] + common_post' * A[:, v] + common_post' * A[:, u] + A[u, :]' * common_pre + A[v, :]' * common_pre
        delta[16] = common_post' * common_pre
    end

    delta = delta * Δuv
    if include_empty
        delta = isomorphism ? over_to_exact_3node * delta : delta
    else
        delta = isomorphism ? over_to_exact_3node[2:16, 2:16] * delta[2:16] : delta[2:16]
    end
end