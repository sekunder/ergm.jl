using ergm
using ergm.models
using ergm.sampling

half_n = 25
n = 2 * half_n

S = [ones(half_n, half_n) zeros(half_n, half_n);
     zeros(half_n, half_n) ones(half_n, half_n)]
for i in 1:n
    S[i,i] = 0
end

M = ScaffoldedEdgeTriangleModel(S, [0.0, -100.0])
sampler = GibbsSampler(M, burn_in=n ^ 3, sample_interval=10 * n^2)
graphs, stats = sample(sampler, 10, progress=true)

stats .* [1 n]