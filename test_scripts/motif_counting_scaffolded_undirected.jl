begin
    using ergm.spaces
    using ergm.models
    using StatsBase
    using LinearAlgebra
    using BenchmarkTools
end

# verifying that motif counting works
begin
    n = 50

    X = randn(n, 3)
    r = 3.0
    innerproducts = X * X'
    Dsquared = (diag(innerproducts) .+ diag(innerproducts)') - 2innerproducts
    S = Dsquared .<= r^2

    A = rand(n, n) .< 0.3
    A[diagind(A)] .= false
    # add a full triangle on nodes 1-3
    for u in 1:3
        for v in 1:3
            u == v && continue
            A[u, v] = true
        end
    end
    rho = sum(A) / (n * (n - 1))
    println("Random graph has $(sum(A)) edges")

    G = ScaffoldedUndirectedGraph(S, true)
    println(G)
    for i = 1:n
        for j = 1:n
            i == j && continue
            G[(i, j)] = A[i, j]
        end
    end
    println(G)
end

begin
    _triplet_codes = [
        1, 2, 2, 3, 2, 4, 6, 8,
        2, 6, 5, 7, 3, 8, 7, 11,
        2, 6, 4, 8, 5, 9, 9, 13,
        6, 10, 9, 14, 7, 14, 12, 15,
        2, 5, 6, 7, 6, 9, 10, 14,
        4, 9, 9, 12, 8, 13, 14, 15,
        3, 7, 8, 11, 7, 12, 14, 15,
        8, 14, 13, 15, 11, 15, 15, 16
    ]
    function exhaustive_count(A::AbstractMatrix)
        brute_force_counts = zeros(Int, 16)
        for u in 1:n
            for v in u + 1:n
                uvvu = (A[u, v] << 1) | A[v, u]
                for w in v + 1:n
                    wvvw = (A[w, v] << 1) | A[v, w]
                    wuuw = (A[w, u] << 1) | A[u, w]
                    c::UInt8 = (wuuw << 4) | (wvvw << 2) | uvvu
                    brute_force_counts[_triplet_codes[c + 1]] += 1
                end
            end
        end
        brute_force_counts
    end

    brute_force_counts = exhaustive_count(G.scaffold_edges)
end

sum(brute_force_counts) == binomial(n, 3) || println("ERROR: brute force counts did not find all triplets")

# include("../src/models/motif_counts.jl")
algebra_counts = triplet_motif_counts(G.scaffold_edges; include_empty=true)

# FIRST TEST: COUNTING TRIPLETS
if all(brute_force_counts .== algebra_counts)
    println("PASS: algebra-powered counting matches brute-force counting")
else
    println("FAIL: mismatch in counting motifs $(findall(brute_force_counts .!= algebra_counts))")
end

# check benchmarks 
# @benchmark brute_force_counts = exhaustive_count(S .* A)
# @benchmark algebra_counts = triplet_motif_counts(S .* A)
# @benchmark brute_force_counts = exhaustive_count(G.scaffold_edges)
# @benchmark algebra_counts = triplet_motif_counts(G.scaffold_edges)

begin
    successes = 0
    trials = 0
    for i in 1:n
        for j in 1:n
            i == j && continue

            original_value = G[(i, j)]
            G[(i, j)] = !original_value
            G[(j, i)] = !original_value

            new_brute_force_counts = exhaustive_count(G.scaffold_edges)
            delta = delta_triplet_motif_counts(G.scaffold_edges, (i, j); include_empty=true)
            algebra_counts = triplet_motif_counts(G.scaffold_edges; include_empty=true)

            if all(new_brute_force_counts .== algebra_counts)
                successes += 1
            else
                println("Mismatch at edge ($i, $j)")
                println("Brute force counts: ", new_brute_force_counts)
                println("Algebra counts: ", algebra_counts)
            end

            # Revert the change
            G[(i, j)] = original_value
            G[(j, i)] = original_value

            trials += 1
        end
    end
end
successes
println("Correctly computed change in stats $(100 * successes / trials) % of the time")
