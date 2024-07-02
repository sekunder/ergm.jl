module inference

include("equilibrium_expectation.jl")
export equilibrium_expectation

include("monte_carlo_gradient_ascent.jl")
export monte_carlo_gradient_ascent, monte_carlo_gradient_ascent_hessian

end