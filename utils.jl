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
    A = max.(0,Z)
    return A, Z
end



"""
    Funtion to initialise the parameters or weights of the desired network.
"""
function initialise_model_weights(layer_dims)
    params = Dict()

    for l=2:length(layer_dims)
        params[string("W_" , string(l-1))] = rand(layer_dims[l] , layer_dims[l-1]) * 0.1
        params[string("b_" , string(l-1))] = zeros(layer_dims[l] , 1)
    end

    return params
end


"""
    Make a linear forward calculation
"""
function linear_forward(A, W, b)
    Z = W*A .+ b
    cache = (A, W, b)

    @assert size(Z) == (size(W, 1), size(A, 2))

    return Z, cache
end


"""
    Make a forward activation from a linear forward.
"""
function linear_forward_activation(A_prev, W, b, activation_function="relu")
    @assert activation_function ∈ ("sigmoid", "relu")

    if activation_function == "sigmoid"
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    end

    if activation_function == "relu"
        Z, linear_cache = linear_forward(A_prev, W, b)
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
    L = length(parameters)/2

    for l = 1: (L-1)
        A_pre = A
        A, cache = linear_forward_activation(A_pre,
                                             parameters[string("W_" , string(Int(l)))],
                                             parameters[string("b_" , string(Int(l)))],
                                             "relu")
        push!(master_cache , cache)
    end

    Ŷ, cache = linear_forward_activation(A,
                                         parameters[string("W_" , string(Int(L)))],
                                         parameters[string("b_" , string(Int(L)))],
                                         "sigmoid")
    push!(master_cache , cache)

    return Ŷ, master_cache

end


"""
    Computes the batch cost of the current predictions
"""
function calculate_cost(Ŷ, Y)
    m = size(Y, 2)
    cost = -sum(Y .* log.(Ŷ) + (1 .- Y) .* log.(1 .- Ŷ))/m
    return cost
end


"""
"""
function sigmoid_backwards(∂A)

end


"""
"""
function relu_backwards(∂A)

end


"""
"""
function linear_backward(∂Z, cache)

end



"""
"""
function linear_backward_activation(∂A, cache, activation_function="relu")

end


"""
    Compute the gradients (∇) of the parameters of the constructed model
    with respect to the cost of predictions.
"""
function back_propagate_model_weights(Ŷ, Y, master_cache)
    ∇ = Dict()

end


"""
    Update the paramaters of the model using the gradients (∇)
    and the learning rate (η).
"""
function update_model_weights(parameters, ∇, η=0.01)

end


"""
"""
function predict(DMatrix, parameters)

end

