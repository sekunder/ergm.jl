using ergm.spaces

mutable struct SimpleModel <: Model
    state::SparseDirectedGraph
    statistics::Vector{Function}
    parameters::Vector{Float64}
    current_statistics::Union{Nothing, Vector{Float64}}

    @doc """
        SimpleModel(number_of_nodes::Int, statistics::Vector{Function}, parameters::Vector{Float64})

    A naively implemented model over sparse directed graphs.
    
    This model should not be used in practice but provides a reference 
    implementation of the Model interface. This model calls the provided
    functions to compute sufficient statistics from scratch every time
    the statistics are queried rather than efficiently updating the
    statistics as the state is incrementally changed and caching them.
    
    # Arguments
    - `number_of_nodes::Int`: define the model on graphs of type SparseDirectedGraph{number_of_nodes}
    - `statistics::Vector`: a vector of functions, each of which takes a `SparseDirectedGraph`
        and returns the value of one of the sufficient statistics.
    - `parameters::Vector{Float64}`: initial values of the natural parameters corresponding
        to each sufficient statistic.
    """
    function SimpleModel(number_of_nodes::Int, statistics::Vector, parameters::Vector{Float64})
        # set state to empty graph to start
        state = SparseDirectedGraph{number_of_nodes}()
        new(state, statistics, parameters, nothing)
    end
end

function get_sample_space(model::SimpleModel)
    typeof(model.state)
end

function get_state(model::SimpleModel)::SparseDirectedGraph
    model.state
end

function set_state(model::SimpleModel, state::SparseDirectedGraph)
    model.state = state

    # invalidate cached statistics
    model.current_statistics = nothing
end

function get_statistics(model::SimpleModel)::Vector{Float64}
    # compute statistics if they are not cached
    if isnothing(model.current_statistics)
        model.current_statistics = [s(model.state) for s in model.statistics]
    end
    
    model.current_statistics
end

function test_update(model::SimpleModel, index::Tuple{Int,Int}, value::Bool)::Vector{Float64}
    updated_state = copy(model.state)
    updated_state[index] = value
    new_statistics = [s(updated_state) for s in model.statistics]
    new_statistics - get_statistics(model)
end

function apply_update(model::SimpleModel, index::Tuple{Int,Int}, value::Bool)
    model.state[index] = value
    
    # invalidate cached statistics
    model.current_statistics = nothing
end
    
function get_parameters(model::SimpleModel)::Vector{Float64}
    model.parameters
end

function set_parameters(model::SimpleModel, parameters::Vector{Float64})
    model.parameters = parameters
end