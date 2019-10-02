
using Gen
using PyPlot
using Random


"""
A simple 1d linear regression model
"""
@gen function line_model(xs::Vector{Float64})

    n = length(xs)

    slope = @trace(normal(0,1), :slope)
    intercept = @trace(normal(0,2), :intercept)

    for (i, x) in enumerate(xs)
        @trace(normal(slope * x + intercept, 0.1), (:y, i) )
    end

    return n

end


"""
A sinusoidal regression model (assumes gaussian noise)
"""
@gen function sine_model(xs::Vector{Number})

    n = length(xs)

    phase = @trace(uniform(0,2*pi), :phase)
    period = @trace(gamma(5,2), :period)
    amplitude = @trace(gamma(2,2), :amplitude)

    for (i, x) in enumerate(xs)
        @trace(normal( amplitude * sin(2*pi*x / period  + phase), 0.1), (:y, i))
    end

    return n

end

"""
Performs importance sampling inference on simple discriminative
models, where vectors of x and y values are provided.
"""
function sample_posterior(model, xs, ys, amount_of_computation)

    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y,i)] = y
    end

    (trace, weight) = Gen.importance_resampling(model, (xs,), observations, amount_of_computation)

    return trace, weight
end

xs = 1.0 .* Vector(-10:0.2:10)

line_ys = 3.14 .* xs .- 1.0 .+ 0.25*randn(length(xs))

sine_ys = 4.0 .* sin.(2.0*pi.*xs ./ 10. .+ 3.0) .+ 1.0*randn(length(xs))
println(sine_ys)
figure(figsize=(10,10))
scatter(xs, sine_ys; color="k")


posterior_line_trace, weight = sample_posterior(line_model, xs, line_ys, 100)

println(posterior_line_trace[:slope])
println(posterior_line_trace[:intercept])

println()
dense_x = Vector(-10:0.1:10)
weights = zeros(Float64, 100)
attrs = zeros(Float64, (100,3))
for i=1:100
    posterior_sine_trace, weight = sample_posterior(sine_model, xs, sine_ys, 200)
    attrs[i, 1] = posterior_sine_trace[:amplitude]
    attrs[i, 2] = posterior_sine_trace[:period]
    attrs[i, 3] = posterior_sine_trace[:phase]
    weights[i] = weight
end

max_weight = maximum(weights)
map!( w -> exp((w - max_weight) / 1000), weights, weights)

for i=1:100
    dense_y = attrs[i,1] .* sin.(2*pi.*dense_x ./ attrs[i,2] .+ attrs[i,3])
    plt.plot(dense_x, dense_y, linewidth=weights[i], color="k")
end

savefig("dumb_figure.png", dpi=200)
