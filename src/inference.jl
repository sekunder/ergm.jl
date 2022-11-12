module inference

using ergm.sampler, ergm.models, ergm.stats, ergm.optim, ergm.spaces
using Statistics
using Random
import StatsBase
using Infiltrator
export mcmc_mle, mcmc_mle_from_stats, ee

function ee(observations, stats, estimation_steps, θ0, iterations, learning_rate)
    observations = shuffle(observations)
    # note that stats will have state set to last observation in shuffled
    # order, effectively initializing the Markov chain with a random observation
    observation_stats = hcat([get_stats(stats, o) for o ∈ observations]...)
    target_Es = mean(observation_stats, dims=2)[:, 1]
    n = observations[1].n
    p = stat_count(stats)
    θ = copy(θ0)
    θs = zeros(iterations, p)
    Es = zeros(iterations, p)
    Ls = zeros(iterations)
    
    # per-parameter learning rate
    D = fill(learning_rate, p)

    println("Starting inference...")

    for it ∈ 1:iterations
        E = zeros(p)
        dt = zeros(p)
        println("iter $it...")

        for _ ∈ 1:estimation_steps
            # propose uniform new edge value for uniform edge
            i = tuple(StatsBase.sample(1:n, 2, replace=false)...)
            old_x = stats.graph[i]
            x = !old_x

            # acceptance probability
            dstats = test_update(stats, (i, x)) - get_stats(stats)
            dll = sum(θ .* dstats)
            α = min(1, exp(dll))

            if rand() < α
                apply_update(stats, (i, x))
            end
            
            E += get_stats(stats)
        end

        dt = get_stats(stats) - target_Es
        θ -= D .* sign.(dt) .* dt .^ 2
        θs[it, :] = θ
        E /= iterations
        Es[it, :] = E
        Ls[it] = sum((E - target_Es) .^ 2) / p
        
        # adapt learning rates every a iterations
        a = 100
        if it % a == 0
            ix = (it - a + 1):it
            θ_m = mean(θs[ix, :], dims=1)[1, :]
            θ_sd = std(θs[ix, :], dims=1)[1, :]
            D .*= sqrt.(1e0 * max.(abs.(θ_m), 1e-2) ./ θ_sd)
        end
    end
    
    θs, Ls, target_Es, Es
end

end
