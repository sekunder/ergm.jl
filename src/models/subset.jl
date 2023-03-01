"""
Simplify an ERGM by fixing all but a subset of its parameters.

# Arguments
- `full_model::Model`: the original model
- `parameter_subset::Vector{Int}`: a list of the indices of parameters
    to retain from the original model. The order of `parameter_subset`
    determines the ordering of parameters in the simplified model.
"""
struct SubsetModel <: Model
    full_model::Model
    parameter_subset::Vector{Int}
end

function get_sample_space(model::SubsetModel)::Any
    get_sample_space(model.full_model)
end

function get_state(model::SubsetModel)::Any
    get_state(model.full_model)
end

function set_state(model::SubsetModel, state::Any)
    set_state(model.full_model, state)
end

function get_statistics(model::SubsetModel)::Vector{Float64}
    full_stats = get_statistics(model.full_model)
    full_stats[model.parameter_subset]
end

function test_update(model::SubsetModel, index::Any, value::Any)::Vector{Float64}
    δfull_stats = test_update(model.full_model, index, value)
    δfull_stats[model.parameter_subset]
end

function apply_update(model::SubsetModel, index::Any, value::Any)
    apply_update(model.full_model, index, value)
end

function get_parameters(model::SubsetModel)::Vector{Float64}
    full_params = get_parameters(model.full_model)
    full_params[model.parameter_subset]
end

function set_parameters(model::SubsetModel, parameters::Vector{Float64})
    full_params = copy(get_parameters(model.full_model))
    full_params[model.parameter_subset] = parameters
    set_parameters(model.full_model, full_params)
end
