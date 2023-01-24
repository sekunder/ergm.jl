using ergm.spaces

mutable struct SimpleModel <: Model
    state :: SparseDirectedGraph
    sufficient_statistics :: Vector{Function}

    @doc """
        SimpleModel(sufficient_statistics :: Vector{Function})

    A naively implemented model over sparse directed graphs.
    
    This model should not be used in practice but provides a reference 
    implementation of the Model interface. This model calls the provided
    functions to compute sufficient statistics from scratch every time
    the statistics are queried rather than efficiently updating the
    statistics as the state is incrementally changed and caching them.
    
    # Arguments
    - `sufficient_statistics :: Vector{Float64}`: a vector of functions, each of
        which takes a `SparseDirectedGraph` and returns the value of some statistic.
    """
    function SimpleModel(sufficient_statistics :: Vector{Function})
        # set state to empty graph to start
        state = SparseDirectedGraph()
    end
end

function get_state(model :: SimpleModel) :: SparseDirectedGraph
    model.state
end

function set_state(model :: SimpleModel, state :: SparseDirectedGraph)
    model.state = state
end

function get_stats(model :: SimpleModel) :: Vector{Float64}
    [s(model.state) for s in model.sufficient_statistics]
end

function test_update(model :: SimpleModel, update :: Tuple{Tuple{Int, Int}, Bool}) :: Vector{Float64}
    edge, value = update
    updated_state = copy(model.state)
    updated_state[edge] = value
    [s(model.state) for s in model.sufficient_statistics]
end

function apply_udate(model :: SimpleModel, update :: Tuple{Tuple{Int, Int}, Bool})
    edge, value = update
    model.state[edge] = value
end