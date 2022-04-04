module optim

using LinearAlgebra, Statistics
export SGD, ADAM, Newton, optim_step, done

mutable struct SGD
    α
    x
    max_iter
    grad_tol
    log_file
    iter
    done

    function SGD(α :: Float64, x0 :: Array{Float64}, max_iter :: Int, grad_tol :: Float64, log_name :: String = nothing)
        if log_name !== nothing
            log_file = open(log_name, "w+")
        else
            log_file = nothing
        end

        new(α, x0, max_iter, grad_tol, log_file, 0, false)
    end
end

function log(o :: SGD, s :: String)
    if o.log_file !== nothing
        write(o.log_file, s)
        flush(o.log_file)
    end
end

function optim_step(o :: SGD, ss :: Matrix{Float64}, target_Es :: Vector{Float64})
    g = (mean(ss, dims=2) - target_Es)[:, 1]
    optim_step(o, g)
end

function optim_step(o :: SGD, g :: Vector{Float64})
    gn = norm(g)

    if o.iter > o.max_iter
        log(o, "Maximum iterations ($(o.max_iter)) reached. Terminating.\n")
        finish(o)
    elseif gn ≤ o.grad_tol
        log(o, "Gradient tolerance (|g| = $gn ≤ $(o.grad_tol)) reached. Terminating.\n")
        finish(o)
    else
        log(o, "Iteration $(o.iter): |g| = $gn.\n")
        o.x -= o.α * g
        o.iter += 1
    end

    o.x
end

function finish(o :: SGD)
    o.done = true
    close(o.log_file)
end

function done(o :: SGD)
    o.done
end

mutable struct ADAM
    α
    β1
    β2
    ε
    x
    m
    v
    max_iter
    grad_tol
    log_file
    iter
    done

    function ADAM(
        α :: Float64, β1 :: Float64, β2 :: Float64, ε :: Float64,
        x0 :: Array{Float64}, max_iter :: Int, grad_tol :: Float64,
        log_name :: String = nothing
    )
        if log_name !== nothing
            log_file = open(log_name, "w+")
        else
            log_file = nothing
        end

        m0 = zeros(size(x0))
        v0 = zeros(size(x0))

        new(
            α, β1, β2, ε,
            x0, m0, v0,
            max_iter, grad_tol, log_file,
            0, false
        )
    end
end

function log(o :: ADAM, s :: String)
    if o.log_file !== nothing
        write(o.log_file, s)
        flush(o.log_file)
    end
end

function optim_step(o :: ADAM, ss :: Matrix{Float64}, target_Es :: Vector{Float64})
    G = ss .- target_Es
    g = mean(G, dims=2)
    gn = norm(g)

    if o.iter > o.max_iter
        log(o, "Maximum iterations ($(o.max_iter)) reached. Terminating.\n")
        finish(o)
    elseif gn ≤ o.grad_tol
        log(o, "Gradient tolerance (|g| = $gn ≤ $(o.grad_tol)) reached. Terminating.\n")
        finish(o)
    else
        log(o, "Iteration $(o.iter): |g| = $gn.\n")
        o.iter += 1
        o.m = o.β1 * o.m + (1 - o.β1) * g
        o.v = o.β2 * o.v + (1 - o.β2) * mean(G .^ 2, dims=2)
        mc = o.m / (1 - o.β1 ^ o.iter)
        vc = o.v / (1 - o.β2 ^ o.iter)
        o.x -= o.α * mc ./ (sqrt.(vc) .+ o.ε)
    end

    o.x
end

function finish(o :: ADAM)
    o.done = true
    close(o.log_file)
end

function done(o :: ADAM)
    o.done
end

mutable struct Newton
    x
    max_iter
    grad_tol
    log_file
    iter
    done

    function Newton(x0 :: Vector{Float64}, max_iter :: Int, grad_tol :: Float64, log_name :: String = nothing)
        if log_name !== nothing
            log_file = open(log_name, "w+")
        else
            log_file = nothing
        end

        new(x0, max_iter, grad_tol, log_file, 0, false)
    end
end

function log(o :: Newton, s :: String)
    if o.log_file !== nothing
        write(o.log_file, s)
        flush(o.log_file)
    end
end

function optim_step(o :: Newton, ss :: Matrix{Float64}, target_Es :: Vector{Float64})
    p, n = size(ss)
    g = mean(ss, dims=2) - target_Es
    gn = norm(g)

    if o.iter > o.max_iter
        log(o, "Maximum iterations ($(o.max_iter)) reached. Terminating.\n")
        finish(o)
    elseif gn ≤ o.grad_tol
        log(o, "Gradient tolerance (|g| = $gn ≤ $(o.grad_tol)) reached. Terminating.\n")
        finish(o)
    else
        log(o, "Iteration $(o.iter): |g| = $gn.\n")
        H = 1 / (n - 1) * ss * ss'
        o.x -= H^-1 * g 
        o.iter += 1
    end

    o.x
end

function finish(o :: Newton)
    o.done = true
    close(o.log_file)
end

function done(o :: Newton)
    o.done
end

end
