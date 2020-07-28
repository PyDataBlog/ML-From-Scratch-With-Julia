include("main.jl")

using HDF5
using Images
using ImageView
using Plots
using MLJBase


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


"""
    Make predictions on new data using the trained paramaters
"""
function predict(DMatrix, parameters)
    Ŷ , _  = forward_propagate_model_weights(DMatrix, parameters)
    return Ŷ
end

#url = "https://raw.githubusercontent.com/PyDataBlog/NN-From-Scratch-With-Julia/master/train_catvnoncat.h5";

# Generate fake data
X, y = make_blobs(10_000, 3; centers=2, as_table=false, rng=2020);
X = Matrix(X');
y = reshape(y, (1, size(X, 2)));
f(x) =  x == 2 ? 0 : x
y2 = f.(y);

# Input dimensions
input_dim = size(X, 1);

train_network([input_dim, 5, 3, 1], X, y2; η=0.01, epochs=50, seed=1, verbose=true);