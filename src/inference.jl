module inference

using ergm.sampler, ergm.models, ergm.stats, ergm.optim, ergm.spaces
using Statistics
using Random
import StatsBase
export ee, cd
using ProgressMeter

function ee(model, initial_state, target_Es, estimation_steps, c2s)
    iterations = length(c2s)
    c1 = 1e-2
    p1 = 2
    p2 = 1/2
    set_state(model.stats, initial_state)
    state = initial_state
    n = model.sample_space.number_of_nodes
    p = stat_count(model.stats)
    θ = model.params
    θs = zeros(iterations, p)
    Es = zeros(iterations, p)
    Ls = zeros(iterations)
    fs = nothing
    
    # per-parameter learning rate
    D = ones(p)
    Ds = nothing

    println("Starting inference...")

    @showprogress for it ∈ 1:iterations
        E = zeros(p)
        dt = zeros(p)

        for _ ∈ 1:estimation_steps
            # propose uniform new edge value for uniform edge
            i = random_index(model.sample_space)
            old_x = state[i]
            x = 1 - old_x

            # acceptance probability
            dstats = test_update(model.stats, (i, x)) - get_stats(model.stats)
            dll = sum(θ .* dstats)
            α = min(1, exp(dll))

            if rand() < α
                state[i] = x
                apply_update(model.stats, (i, x))
            end
            
            E += get_stats(model.stats)
        end

        dt = get_stats(model.stats) - target_Es
        θ -= D .* sign.(dt) .* dt .^ p1
        θs[it, :] = θ
        E /= estimation_steps
        Es[it, :] = E
        Ls[it] = sum((E - target_Es) .^ 2) / p
        
        # adapt learning rates every a iterations
        a = 500

        if it % a == 0
            c2 = c2s[it]
            ix = (it - a + 1):it
            θ_m = mean(θs[ix, :], dims=1)[1, :]
            θ_sd = std(θs[ix, :], dims=1)[1, :]
            D .*= (c2 * max.(abs.(θ_m), c1) ./ θ_sd) .^ p2
            f = θ_sd ./ (c2 * max.(abs.(θ_m), c1))

            if fs == nothing
                fs = f'
                Ds = D'
            else
                fs = vcat(fs, f')
                Ds = vcat(Ds, D')
            end
        end
    end
    
    θs, Ls, target_Es, Es, fs, Ds
end

function cd(model, initial_state, target_Es, estimation_steps, iterations)
    c1 = 1e-2
    c2 = 1e-3
    p1 = 2
    p2 = 1/2
    set_state(model.stats, initial_state)
    state = initial_state
    n = model.sample_space.number_of_nodes
    p = stat_count(model.stats)
    θ = model.params
    θs = zeros(iterations, p)
    Es = zeros(iterations, p)
    Ls = zeros(iterations)
    fs = nothing
    
    # per-parameter learning rate
    D = ones(p)
    Ds = nothing

    println("Starting inference...")

    initial_stats = model.stats

    for it ∈ 1:iterations
        E = zeros(p)
        dt = zeros(p)

        state = copy(initial_state)
        model.stats = copy(initial_stats)

        dzsum = zeros(p)

        for _ ∈ 1:estimation_steps
            # propose uniform new edge value for uniform edge
            i = random_index(model.sample_space)
            old_x = state[i]
            x = 1 - old_x

            # acceptance probability
            dstats = test_update(model.stats, (i, x)) - get_stats(model.stats)
            dll = sum(θ .* dstats)
            α = min(1, exp(dll))

            if rand() < α
                dzsum .+= abs.(dstats)
                state[i] = x
                apply_update(model.stats, (i, x))
            end
            
            E += get_stats(model.stats)
        end

        dt = get_stats(model.stats) - target_Es
        println(dzsum)
        da = 10 ./ dzsum .^ 2
        θ -= da .* sign.(dt) .* dt .^ p1
        θs[it, :] = θ
        E /= estimation_steps
        Es[it, :] = E
        Ls[it] = sum((E - target_Es) .^ 2) / p
        
        # adapt learning rates every a iterations
        # a = 1000
        # if it % a == 0
        #     ix = (it - a + 1):it
        #     θ_m = mean(θs[ix, :], dims=1)[1, :]
        #     θ_sd = std(θs[ix, :], dims=1)[1, :]
        #     D .*= (c2 * max.(abs.(θ_m), c1) ./ θ_sd) .^ p2
        #     f = θ_sd ./ (c2 * max.(abs.(θ_m), c1))

        #     if fs == nothing
        #         fs = f'
        #         Ds = D'
        #     else
        #         fs = vcat(fs, f')
        #         Ds = vcat(Ds, D')
        #     end
        # end
    end
    
    θs, Ls, target_Es, Es, fs, Ds
end

end
