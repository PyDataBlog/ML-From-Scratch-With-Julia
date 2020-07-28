# Import seeding generator for reproducibility
using StableRNGs


"""
    Sigmoid activation function
"""
function sigmoid(Z)
    A = 1 ./ (1 .+ exp.(.-Z))
    return A, Z
end


"""
    ReLU activation function
"""
function relu(Z)
    A = max.(0, Z)
    return A, Z
end


"""
    Funtion to initialise the parameters or weights of the desired network.
"""
function initialise_model_weights(layer_dims, seed)
    params = Dict()

    # Build a dictionary of initialised weights and bias units
    for l=2:length(layer_dims)
        params[string("W_", string(l-1))] = rand(StableRNG(seed), layer_dims[l], layer_dims[l-1]) * sqrt(2 / layer_dims[l-1])
        params[string("b_", string(l-1))] = zeros(layer_dims[l], 1)
    end

    return params
end


"""
    Make a linear forward calculation
"""
function linear_forward(A, W, b)
    # Make a linear forward and return inputs as cache
    Z = (W * A) .+ b
    cache = (A, W, b)

    @assert size(Z) == (size(W, 1), size(A, 2))

    return Z, cache
end


"""
    Make a forward activation from a linear forward.
"""
function linear_forward_activation(A_prev, W, b, activation_function="relu")
    @assert activation_function ∈ ("sigmoid", "relu")
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation_function == "sigmoid"
        A, activation_cache = sigmoid(Z)
    end

    if activation_function == "relu"
        A, activation_cache = relu(Z)
    end

    cache = (linear_step_cache=linear_cache, activation_step_cache=activation_cache)

    @assert size(A) == (size(W, 1), size(A_prev, 2))

    return A, cache
end


"""
    Forward the design matrix through the network layers using the parameters.
"""
function forward_propagate_model_weights(DMatrix, parameters)
    master_cache = []
    A = DMatrix
    L = Int(length(parameters) / 2)

    for l = 1 : (L-1)
        A_prev = A
        A, cache = linear_forward_activation(A_prev,
                                             parameters[string("W_", string(l))],
                                             parameters[string("b_", string(l))],
                                             "relu")
        push!(master_cache , cache)
    end

    Ŷ, cache = linear_forward_activation(A,
                                         parameters[string("W_", string(L))],
                                         parameters[string("b_", string(L))],
                                         "sigmoid")
    push!(master_cache , cache)

    return Ŷ, master_cache

end


"""
    Computes the log loss of the current predictions.
"""
function calculate_cost(Ŷ, Y)
    m = size(Y, 2)
    cost = -sum(Y .* log.(Ŷ) + (1 .- Y) .* log.(1 .- Ŷ)) / m
    return cost
end


"""
    Derivative of the Sigmoid function.
"""
function sigmoid_backwards(∂A, activated_cache)
    s = sigmoid(activated_cache)[1]
    ∂Z = ∂A .* s .* (1 .- s)

    @assert (size(∂Z) == size(activated_cache))
    return ∂Z
end


"""
    Derivative of the ReLU function.
"""
function relu_backwards(∂A, activated_cache)
    return ∂A .* (activated_cache .> 0)
end


"""
    Partial derivatives of the linear forward function.
"""
function linear_backward(∂Z, cache)
    A_prev , W , b = cache
    m = size(A_prev, 2)

    ∂W = ∂Z * (A_prev') / m
    ∂b = sum(∂Z, dims = 2) / m
    ∂A_prev = (W') * ∂Z

    @assert (size(∂A_prev) == size(A_prev))
    @assert (size(∂W) == size(W))
    @assert (size(∂b) == size(b))

    return ∂W , ∂b , ∂A_prev
end


"""
    Forward the design matrix through the network layers using the parameters.
"""
function linear_activation_backward(∂A, cache, activation_function="relu")
    @assert activation_function ∈ ("sigmoid", "relu")

    linear_cache , cache_activation = cache

    if (activation_function == "relu")

        ∂Z = relu_backwards(∂A , cache_activation)
        ∂W , ∂b , ∂A_prev = linear_backward(∂Z , linear_cache)

    elseif (activation_function == "sigmoid")

        ∂Z = sigmoid_backwards(∂A , cache_activation)
        ∂W , ∂b , ∂A_prev = linear_backward(∂Z , linear_cache)

    end

    return ∂W , ∂b , ∂A_prev
end


"""
    Compute the gradients (∇) of the parameters of the constructed model
    with respect to the cost of predictions.
"""
function back_propagate_model_weights(Ŷ, Y, master_cache)
    ∇ = Dict()

    L = length(master_cache)

    m = size(Ŷ, 2)
    Y = reshape(Y , size(Ŷ))

    ∂Ŷ = (-(Y ./ Ŷ) .+ ((1 .- Y) ./ ( 1 .- Ŷ)))
    current_cache = master_cache[L]

    ∇[string("∂W_", string(L))], ∇[string("∂b_", string(L))], ∇[string("∂A_", string(L-1))] = linear_activation_backward(∂Ŷ,
                                                                                                                         current_cache,
                                                                                                                        "sigmoid")
    for l=reverse(0:L-2)
        current_cache = master_cache[l+1]

        ∇[string("∂W_", string(l+1))], ∇[string("∂b_", string(l+1))], ∇[string("∂A_", string(l))] = linear_activation_backward(
                                                                                                                ∇[string("∂A_", string(l+1))],
                                                                                                                current_cache,
                                                                                                                "relu")
    end

    return ∇
end


"""
    Update the paramaters of the model using the gradients (∇)
    and the learning rate (η).
"""
function update_model_weights(parameters, ∇, η)

    L = Int(length(parameters) / 2)

    for l = 0: (L-1)
        parameters[string("W_" , string(l + 1))] -= η .* ∇[string("∂W_" , string(l + 1))]
        parameters[string("b_", string(l + 1))] -= η .* ∇[string("∂b_",string(l + 1))]
    end

    return parameters
end


"""
    Check the accuracy between predicted values (Ŷ) and the true values(Y).
"""
function assess_accuracy(Ŷ , Y)
    @assert size(Ŷ) == size(Y)
    return sum((Ŷ .> 0.5) .== Y) / length(Y)
end


"""
    Train the network using the desired architecture that best possible
    matches the training inputs (DMatrix) and their corresponding ouptuts(Y)
    over some number of iterations (epochs) and a learning rate (η).
"""
function train_network(layer_dims , DMatrix, Y;  η=0.001, epochs=1000, seed=2020, verbose=true)
    costs = []
    iters = []
    accuracy = []

    params = initialise_model_weights(layer_dims, seed)

    for i = 1:epochs

        Ŷ , caches  = forward_propagate_model_weights(DMatrix, params)
        cost = calculate_cost(Ŷ, Y)
        acc = assess_accuracy(Ŷ, Y)
        ∇  = back_propagate_model_weights(Ŷ, Y, caches)
        params = update_model_weights(params, ∇, η)

        if verbose
            println("Iteration -> $i, Cost -> $cost, Accuracy -> $acc")
        end

        push!(iters , i)
        push!(costs , cost)
        push!(accuracy , acc)
    end
        return (cost = costs, iterations = iters, accuracy = accuracy, parameters = params)
end


