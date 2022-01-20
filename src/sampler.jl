module sampler

mutable struct GibbsSampler{T}
    initial_state :: T
    state :: T
    ll_calculator
    burn_in :: Int
    sample_interval :: Int
    
    function GibbsSampler(
        initial_state :: T,
        ll_calculator,
        burn_in :: Int,
        sample_interval :: Int
    ) where {T}
        new{T}(initial_state, initial_state, ll_calculator, burn_in, sample_interval)
    end
end

function gibbs_step(sampler :: GibbsSampler{T}) :: T where {T}
    # is updating components in a fixed order statistically valid?
    for i ∈ eachindex(sampler.state)
        sampler.statistic_calculator()
    end
end

function sample(sampler :: GibbsSampler{T}, n :: Int) :: Vector{T} where {T}
    sampler.state = sampler.initial_state
    samples = Vector{T}(undef, n)
    
    # burn in to reach equilibrium state
    for _ ∈ 1:sampler.burn_in
        gibbs_step(sampler)
    end

    for _ ∈ 1:n
        # draw one sample
        samples[i] = gibbs_step(sampler)
        
        # throw away some samples to reduce autocorrelation
        for _ ∈ sampler.sample_interval
            gibbs_step(sampler)
        end
    end
    
    samples
end

end