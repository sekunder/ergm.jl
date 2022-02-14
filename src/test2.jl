using ergm.optim

f(x) = x[1]^2 + x[2]^4
g(x) = [2x[1], 4x[2]^3]
x = [1.0, 2.0]
s = ADAM(1.0, 0.9, 0.999, 1e-8, x, 1000, 0.001, "a")
#s = SGD(0.1, x, 1000, 0.001, "a")

while !done(s)
    global x = optim_step(s, g(x))
end

println(x)
