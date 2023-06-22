using ergm.spaces
using DataStructures
using SparseArrays
using LinearAlgebra

const exact_to_over_3node  = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1; # 1 empty graph
                              0 1 2 2 2 2 3 3 3 3 4 4 4 4 5 6; # 2 single edge
                              0 0 1 0 0 0 1 1 0 0 2 1 1 1 2 3; # 3 single reciprocal
                              0 0 0 1 0 0 0 1 1 0 1 1 2 1 2 3; # 4 diverging
                              0 0 0 0 1 0 1 0 1 0 1 2 1 1 2 3; # 5 converging
                              0 0 0 0 0 1 1 1 1 3 2 2 2 3 4 6; # 6 two step path
                              0 0 0 0 0 0 1 0 0 0 2 2 0 1 3 6; # 7 single edge into a reciprocal
                              0 0 0 0 0 0 0 1 0 0 2 0 2 1 3 6; # 8 single edge ouf of a reciprocal
                              0 0 0 0 0 0 0 0 1 0 0 2 2 1 3 6; # 9 directed clique, diverging to a single edge
                              0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 2; # 10 directed cycle
                              0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 3; # 11 reciprocal path
                              0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 3; # 12 diverging to a reciprocal
                              0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 3; # 13 converging from a reciprocal
                              0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 6; # 14 path across a reciprocal
                              0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 6; # 15 all but one edge
                              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1] # 16 all six edges
const over_to_exact_3node = Matrix{Int}(inv(exact_to_over_3node))

"""
    triplet_motif_counts(A::Matrix, isomorphism=True, include_empty=False)

Compute the number of occurrences of each isomorphism class of three-node graphs in A.

By default, counts subgraph _isomorphisms_, that is, the number of _induced_ copies of each motif.
Also by default, returns a vector of length 15, which does _not_ count the empty triad, but it
will count the single-edge and reciprocal-edge motifs.

# Arguments
- `A` adjacency matrix for directed graph. `A[i,j]` indicates the present of edge `i -> j`
- `isomorphism` Whether to count subgraph isomorphisms (`True`) or subgraph monomorphisms (`False`)
"""
function triplet_motif_counts(A::Matrix, isomorphism=True)
    n1, n2 = size(A)
    n1 == n2 || error("Matrix must be square")
    n = n1
    m = sum(A)  # number of edges

    counts = zeros(15)  # we'll put in the empty count at the end, if needed
    A2 = A ^ 2
    trA2 = tr(A2)

    counts[1] = (n - 2) * m  # single edge + disconnected third node
    counts[2] = (n - 2) * trA2 ÷ 2  # single reciprocal connection + disconnected third node
    counts[5] = sum(A2) - trA2  # sum off-diagonal elements of A^2, counts two-step paths

    diverging = A' * A  # common inputs
    converging = A * A'  # common outputs
    counts[3] = (sum(diverging) - m) ÷ 2 
    counts[4] = (sum(converging) - m) ÷ 2
    counts[9] = sum(A .* A2)

    A3 = A ^ 3
    counts[10] = tr(A3) ÷ 3

    U = A .* A'  # reciprocal connections only
    D = sum(U; dims=2)
    counts[8] = sum(A .* D) - sum(U)
    counts[7] = sum(A .* D') - sum(U)
    counts[13] = sum(U .* converging) ÷ 2
    counts[14] = sum(U .* A2)
    counts[12] = sum(U .* diverging) ÷ 2

    U2 = U ^ 2
    counts[11] = (sum(U2) - tr(U2)) ÷ 2
    counts[14] = sum(U2 .* U)
    counts[15] = tr(U2 * U) ÷ 6

    counts = isomorphism ? over_to_exact_3node[2:16, 2:16] * counts : counts
end


"""
    delta_triplet_motif_counts(A::Matrix, index, isomorphism=True)

Compute change in motif counts if edge `index` is toggled
"""
function delta_triplet_motif_counts(A::Matrix, index, isomorphism=True)
    n1, n2 = size(A)
    n1 == n2 || error("Matrix must be square")
    n = n1
    m = sum(A)
    u,v = index
    Duv = 2 * A[u,v] - 1

    delta = zeros(15)
    delta[1] = (n - 2)
    delta[2] = (n - 2) * A[v, u]
    # STILL WORKING ON THIS
end