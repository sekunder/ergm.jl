module models

# define general interface for models
abstract type Model end

"""
    get_state(model :: Model) :: Any

Retrieve the current model state.
"""
function get_state(model::Model)::Any
    error("unimplemented")   
end

"""
    set_state(model :: Model, state :: Any)

Set the model's state to a particular graph. This will throw away all saved information about the
previous state and start computing sufficient statistics from scratch for the new state.
"""
function set_state(model::Model, state::Any)
    error("unimplemented")   
end

"""
    get_stats(model :: Model) :: Vector{Float64}

Return a vector of the sufficient statistics of the current model state.
"""
function get_stats(model::Model)::Vector{Float64}
    error("unimplemented")   
end

"""
    test_update(update :: Any) :: Vector{Float64}

Compute the new sufficient statistics if we were to apply an update to the current state.

Note that this function does not actually change the model state. The form of an update
depends on the particular model. Usually, an update will look like `((i, j), x)`, which
corresponds to updating the value of the edge between nodes `i` and `j` to be `x`
(for example, `x` can be `true` or `false` for unweighted graphs).
"""
function test_update(update::Any)::Vector{Float64}
    error("unimplemented")   
end

"""
    apply_update(update :: Any)

Apply an update the model state.

In contrast to `test_update`, this function will actually mutate the model state. This
function does not return the new sufficient statistics, since they can now be computed
using `get_stats` because the model state was updated.
"""
function apply_update(update::Any)
    error("unimplemented")   
end

export Model, get_state, set_state, get_stats, test_update, apply_update

# export pre-defined models
include("simple.jl")
export SimpleModel
include("directed_spatial_triplet.jl")
export DirectedSpatialTripletModel

end