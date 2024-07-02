begin
    # using ergm.spaces
    using ergm.models
    using StatsBase
    using LinearAlgebra
    using BenchmarkTools
end



# verifying that motif counting works
begin
    n = 50
    # X = randn(n, 3)
    # r = 3.0
    # innerproducts = X * X'
    # Dsquared = (diag(innerproducts) .+ diag(innerproducts)') - 2innerproducts
    # A = Dsquared .<= r^2
    A = rand(n,n) .< 0.3
    A[diagind(A)] .= false
    # add a full triangle on nodes 1-3
    for u = 1:3
        for v = 1:3
            u == v && continue
            A[u,v] = true
        end
    end
    rho = sum(A) / (n * (n-1))
end

begin
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
function exhaustive_count(A::AbstractMatrix)
    brute_force_counts = zeros(Int, 16)
    for u in 1:n
        for v in u+1:n
            uvvu = (A[u,v] << 1) | A[v,u]
            for w in v+1:n
                wvvw = (A[w,v] << 1) | A[v,w]
                wuuw = (A[w,u] << 1) | A[u,w]
                c::UInt8 = (wuuw << 4) | (wvvw << 2) | uvvu
                brute_force_counts[_triplet_codes[c+1]] += 1
            end
        end
    end
    brute_force_counts
end

brute_force_counts = exhaustive_count(A)
end

sum(brute_force_counts) ==  binomial(n, 3)

# include("../src/models/motif_counts.jl")
algebra_counts = triplet_motif_counts(A; include_empty=true)

# FIRST TEST: COUNTING TRIPLETS
if all(brute_force_counts .== algebra_counts)
    println("PASS: algebra-powered counting matches brute-force counting")
else
    println("FAIL: mismatch in counting motifs $(findall(brute_force_counts .!= algebra_counts))")
end

# check benchmarks 
# @benchmark brute_force_counts = exhaustive_count(A)
# @benchmark algebra_counts = triplet_motif_counts(A)

# now change several edges and see what happens
# T = 100
# successes = 0
# for t in 1:T
    # u,v = sample(1:n, 2; replace=false)
    # u,v = 1,2  # toggle an edge I *know* is involved in a full triplet
    # delta = delta_triplet_motif_counts(A, (u,v); include_empty=true)
    # algebra_counts = algebra_counts + delta
    # Acopy = copy(A)
    # Acopy[u,v] = !Acopy[u,v]
    # new_brute_force_counts = exhaustive_count(Acopy)
    # new_algebra_counts = triplet_motif_counts(Acopy; include_empty=true)

    # # SECOND TEST: COUNTING CHANGES
    # # successes += all(brute_force_counts[2:end] .== algebra_counts)
    # new_brute_force_counts .== algebra_counts
    # all(new_brute_force_counts .== new_algebra_counts)
    # (new_brute_force_counts - brute_force_counts) .== delta

    # delta_mono_algebra = exact_to_over_3node * delta
    # delta_mono_brute = exact_to_over_3node * (new_brute_force_counts - brute_force_counts)
    # delta_mono_algebra .== delta_mono_brute
    # findall(delta_mono_algebra .!= delta_mono_brute)
# end
# println("Successfully counted motif change $(100 * successes / T)% of the time")
begin
successes = 0
trials = 0
for i in 1:n
    for j in 1:n
        i == j && continue
        delta = delta_triplet_motif_counts(A, (i, j); include_empty=true)
        algebra_counts = algebra_counts + delta
        A[i,j] = !A[i,j]
        brute_force_counts = exhaustive_count(A)
        successes += all(algebra_counts .== brute_force_counts)
        trials += 1
    end
end
end
println("Correctly computed change in stats $(100 * successes / trials) % of the time")