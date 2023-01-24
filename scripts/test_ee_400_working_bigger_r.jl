using ergm.stats
using ergm.models
using ergm.sampler
using ergm.spaces
using ergm.optim
using ergm.inference
using Statistics
using LinearAlgebra
using SparseArrays
using Plots

# ground-truth
n = 1000
Ω = SparseGraphs(n)
X = 2 * rand(n, 3)
r = 0.65
θ_gt = [-1e2, 4e1]
model = ERGM(Ω, LocalEdgeTriangle, θ_gt, X, r)
sampler = GibbsSampler(model, burn_in=5*n^2, sample_interval=n^2)
Gs, ss = sample(sampler, 1)

target_Es = mean(ss, dims=2)[:, 1]

# sample E-R with same edge density
p = ss[1] / (n - 1) * 2
Gs_ER = [SparseGraph(convert(SparseMatrixCSC{Int64, Int64}, sprand(Bool, n, n, p))) for _ ∈ 1:100]
ss_er = mean(hcat([get_stats(model.stats, G) for G ∈ Gs_ER]...), dims=2)

# how over-represented are local triangles
println(ss[2] / ss_er[2]) # -> 3.78
 
function nlt(G)
    get_stats(EdgeTriangle(Ω), G)[2] - get_stats(model.stats, G)[2]
end

# how over-represented are non-local triangles (excludes local triangles)
println(nlt(Gs[1]) / mean(nlt.(Gs_ER))) # -> 0.997

function plt(θ_gt, θs)
    plot(θs[:, 1], θs[:, 2])
    scatter!([θ_gt[1]], [θ_gt[2]], color=:red)
end

# fitting
model.params = [0e0, 0e0]
c2s = [fill(5e-2, 10000); fill(5e-3, 10000); fill(1e-3, 10000); fill(5e-4, 10000); fill(1e-4, 50000)]
θs, Ls, target_Es, Es, fs, Ds = ee(model, Gs[1], target_Es, 100, c2s)
