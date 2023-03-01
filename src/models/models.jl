module models

# define general interface for models
abstract type Model end

"""
    get_sample_space(model::Model)::Any

Retrieve the sample space over which the model is defined.
"""
function get_sample_space(model::Model)::Any
    error("unimplemented")   
end

"""
    get_state(model::Model)::Any

Retrieve the current model state.
"""
function get_state(model::Model)::Any
    error("unimplemented")   
end

"""
    set_state(model::Model, state::Any)

Set the model's state to a particular graph. This will throw away all saved information about the
previous state and start computing sufficient statistics from scratch for the new state.
"""
function set_state(model::Model, state::Any)
    error("unimplemented")   
end

"""
    get_statistics(model::Model)::Vector{Float64}

Return a vector of the sufficient statistics of the current model state.
"""
function get_statistics(model::Model)::Vector{Float64}
    error("unimplemented")   
end

"""
    test_update(model::Model, index::Any, value::Any)::Vector{Float64}

Compute the new sufficient statistics if we were to update one edge of the current state graph.

Note that this function does not actually change the model state, but returns the hypothetical new 
statistics if the edge with a given index is set to the given value
"""
function test_update(model::Model, index::Any, value::Any)::Vector{Float64}
    error("unimplemented")   
end

"""
    apply_update(model::Model, index::Any, value::Any)

Apply an update the model state.

In contrast to `test_update`, this function will actually mutate the model state. This
function does not return the new sufficient statistics, since they can now be computed
using `get_stats` because the model state was updated.
"""
function apply_update(model::Model, index::Any, value::Any)
    error("unimplemented")   
end

"""
    get_parameters(model::Model)::Vector{Float64}

Retrieve the current natural parameters corresponding to each sufficient statistic.
"""
function get_parameters(model::Model)::Vector{Float64}
    error("unimplemented")
end

"""
    set_parameters(model::Model, parameters::Vector{Float64})

Set the current natural parameters corresponding to each sufficient statistic.
"""
function set_parameters(model::Model, parameters::Vector{Float64})
    error("unimplemented")
end

export Model, get_sample_space, get_state, set_state, get_statistics, test_update, apply_update, get_parameters, set_parameters

# export pre-defined models
include("simple.jl")
export SimpleModel
include("subset.jl")
export SubsetModel
include("directed_spatial_triplet.jl")
export DirectedSpatialTripletModel

end
