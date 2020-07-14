# Import libraries
using Plots
using MLJBase
using HDF5
using Images
using ImageView
using Statistics


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
        params[string("W_" , string(l-1))] = rand(layer_dims[l], layer_dims[l-1]) * 0.1
        params[string("b_" , string(l-1))] = zeros(layer_dims[l], 1)
    end

    return params
end


"""
    Make a linear forward calculation
"""
function linear_forward(A, W, b)
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
    L = Int(length(parameters)/2)

    for l = 1 : (L-1)
        A_prev = A
        A, cache = linear_forward_activation(A_prev,
                                             parameters[string("W_" , string(l))],
                                             parameters[string("b_" , string(l))],
                                             "relu")
        push!(master_cache , cache)
    end

    Ŷ, cache = linear_forward_activation(A,
                                         parameters[string("W_" , string(L))],
                                         parameters[string("b_" , string(L))],
                                         "sigmoid")
    push!(master_cache , cache)

    return Ŷ, master_cache

end


"""
    Computes the batch cost of the current predictions
"""
function calculate_cost(Ŷ, Y)
    m = size(Y, 2)
    cost = -mean(Y .* log.(Ŷ) + (1 .- Y) .* log.(1 .- Ŷ))
    return cost
end


"""
    Derivative of the Sigmoid function
"""
function sigmoid_backwards(∂A, activated_cache)
    s = sigmoid(activated_cache)[1]
    ∂Z = ∂A .* s .* (1 .- s)

    @assert (size(∂Z) == size(activated_cache))
    return ∂Z
end


"""
    Derivative of the ReLU function
"""
function relu_backwards(∂A, activated_cache)
    return ∂A .* (activated_cache .>0)
end


"""
    Partial derivatives of the linear forward function
"""
function linear_backward(∂Z, cache)
    A_prev , W , b = cache
    m = size(A_prev, 2)

    ∂W = ∂Z * (A_prev')/m
    ∂b = sum(∂Z , dims = 2)/m
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

    ∇[string("dW_" , string(L))], ∇[string("db_" , string(L))], ∇[string("dA_" , string(L-1))] = linear_activation_backward(∂Ŷ,
                                                                                                                            current_cache,
                                                                                                                            "sigmoid")
    for l=reverse(0:L-2)
        current_cache = master_cache[l+1]

        ∇[string("dW_", string(l+1))] , ∇[string("db_", string(l+1))] , ∇[string("dA_", string(l))] = linear_activation_backward(
                                                                                                                ∇[string("dA_", string(l+1))],
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

    L = Int(length(parameters)/2)

    for l = 0: (L-1)
        parameters[string("W_" , string(l + 1))] -= η .* ∇[string("dW_" , string(l + 1))]
        parameters[string("b_", string(l + 1))] -= η .* ∇[string("db_",string(l + 1))]
    end
    return parameters
end


"""
    Check the accuracy between predicted values and the true values.
"""
function assess_accuracy(Ŷ , Y)
    @assert size(Ŷ) == size(Y)
    return sum((Ŷ .> 0.5) .== Y) / length(Y)
end


function check_accuracy(A_L , Y)
    A_L = reshape(A_L , size(Y))
    return sum((A_L .> 0.5) .== Y) / length(Y)
end


"""
    Train the network
"""
function train_network(layer_dims , DMatrix, Y,  η=0.001, max_iters=1000)
    costs = []
    iters = []
    accuracy = []

    params = initialise_model_weights(layer_dims)


    for i = 1:max_iters

        Ŷ , caches  = forward_propagate_model_weights(DMatrix, params)
        cost = calculate_cost(Ŷ , Y)
        acc = assess_accuracy(Ŷ , Y)
        #acc = check_accuracy(Ŷ , Y)
        ∇  = back_propagate_model_weights(Ŷ , Y , caches)
        params = update_model_weights(params , ∇ , η)

        println("Iteration -> $i, Cost -> $cost, Accuracy -> $acc.")

        push!(iters , i)
        push!(costs , cost)
        push!(accuracy , acc)
    end
        return (cost=costs, iterations=iters, accuracy=accuracy, parameters=params)
end


"""
    Make predictions on new data using the trained paramaters
"""
function predict(DMatrix, parameters)
    Ŷ , _  = forward_propagate_model_weights(DMatrix, parameters)
    return Ŷ
end


"""
    Function to load flattened toy data to test the implementation.
"""
function load_data()
    X_train = Float64.(h5read("train_catvnoncat.h5", "train_set_x"))
    #X_train = reshape(X_train, :, size(X_train, 4))

    y_train = Int64.(h5read("train_catvnoncat.h5", "train_set_y"))
    #y_train = reshape(y_train, (1, size(y_train, 1)))

    X_test = Float64.(h5read("test_catvnoncat.h5", "test_set_x"))
    #X_test = reshape(X_test, :, size(X_test, 4))

    y_test = Int64.(h5read("test_catvnoncat.h5", "test_set_y"))
    #y_test = reshape(y_test, (1, size(y_test, 1)))

    categories = h5read("test_catvnoncat.h5", "list_classes")

    return (TrainingInputs = X_train, ValidationInputs = X_test,
            TrainTarget = y_train, ValidationTarget = y_test,
            categories = categories)
end


# Generate fake data
X, y = make_blobs(10_000, 3; centers=2, as_table=false, rng=2020)
X = Matrix(X')
y = reshape(y, (1, 10_000))


#url = "https://raw.githubusercontent.com/PyDataBlog/NN-From-Scratch-With-Julia/master/train_catvnoncat.h5";
function replace_2(x)
    if x == 2
        return 0
    else
        return x
    end
end

y2 = replace_2.(y)
#train_network([3, 5, 3, 1], X, y2, 0.3, 300)