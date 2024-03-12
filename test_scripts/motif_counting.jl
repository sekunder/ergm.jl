using ergm.spaces
using ergm.models
using StatsBase
using LinearAlgebra

# verifying that motif counting works
begin
    n = 50
    # X = randn(n, 3)
    # r = 3.0
    # innerproducts = X * X'
    # Dsquared = (diag(innerproducts) .+ diag(innerproducts)') - 2innerproducts
    # A = Dsquared .<= r^2
    # A[diagind(A)] .= false
    A = rand(n,n) .< 0.1
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

algebra_counts = triplet_motif_counts(A)

sum(brute_force_counts) ==  binomial(n, 3)

brute_force_counts[2:end] .== algebra_counts