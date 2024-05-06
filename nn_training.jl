# Activate the environment
using Pkg
Pkg.activate(".")

# Load the packages
using Flux
using JSON
using HDF5
using JuMP
using Plots
using Ipopt
using Mosek
using Random
using Gurobi
using Statistics
using MosekTools
using PowerModels
using SparseArrays
using LinearAlgebra
using InvertedIndices
using BSON: @save, @load

# Add function files
include("src/power_system_utilities.jl")
include("src/model_structures.jl")
include("src/data_collection.jl")
include("src/verification.jl")
include("src/training.jl")

# Define relevant folders
data_folder  = "./data/"
model_folder = "./nn_models/"

# set seed and initialize plotly
Random.seed!(1);
plotly()

# Define case
case = 118

# Call case files and hyperparameters
if case == 14
    # Define files
    pglib_case_name        = "pglib_opf_case14_ieee.m"
    powerflow_data_file    = data_folder*"case14_data.h5"
    powerflow_network_file = data_folder*"case14_network.h5"
    nn_parameter_file      = model_folder*"case14_neural_network.h5"
    BSON_file              = model_folder*"case14_BSON.bson"

    # Hyperparameters
    hyperparams = Dict(
                :num_ReLU                => 30,
                :dense_epoch_iterations  => 30000,
                :sparse_epoch_iterations => 30000,
                :adam                    => true,
                :dense_learning_rate     => 1e-4,
                :sparse_learning_rate    => 1e-4,
                :decay_tup               => (0.9,0.999),
                :print_val               => 200,
                :load_model              => false,
                :batch_size              => 10,
                :shuffle_data            => true,
                :percent_sparse          => 50/100,
                :mip_gap                 => 0.1/100,
                :time_lim                => 60.0,
                :dont_scale_std          => true,
                :setzero_threshold       => 2.5e-3)
elseif case == 57
    # Define files
    pglib_case_name        = "pglib_opf_case57_ieee.m"
    powerflow_data_file    = data_folder*"case57_data.h5"
    powerflow_network_file = data_folder*"case57_network.h5"
    nn_parameter_file      = model_folder*"case57_neural_network.h5"
    BSON_file              = model_folder*"case57_BSON.bson"

    # Hyperparameters
    hyperparams = Dict(
                :num_ReLU                => 60,
                :dense_epoch_iterations  => 30000,
                :sparse_epoch_iterations => 30000,
                :adam                    => true,
                :dense_learning_rate     => 2e-4,
                :sparse_learning_rate    => 2e-4,
                :decay_tup               => (0.9,0.999),
                :print_val               => 200,
                :load_model              => false,
                :batch_size              => 50,
                :shuffle_data            => true,
                :percent_sparse          => 50/100,
                :mip_gap                 => 0.1/100,
                :time_lim                => 60.0,
                :dont_scale_std          => true,
                :setzero_threshold       => 1e-3)

elseif case == 118
    # Define files
    pglib_case_name        = "pglib_opf_case118_ieee.m"
    powerflow_data_file    = data_folder*"case118_data.h5"
    powerflow_network_file = data_folder*"case118_network.h5"
    nn_parameter_file      = model_folder*"case118_neural_network.h5"
    BSON_file              = model_folder*"case118_BSON.bson"

    # Hyperparameters
    hyperparams = Dict(
                :num_ReLU                => 75,
                :dense_epoch_iterations  => 15000,
                :sparse_epoch_iterations => 15000,
                :adam                    => true,
                :dense_learning_rate     => 1e-3,
                :sparse_learning_rate    => 1e-3,
                :decay_tup               => (0.9,0.999),
                :print_val               => 200,
                :load_model              => false,
                :batch_size              => 50,
                :shuffle_data            => true,
                :percent_sparse          => 75/100,
                :mip_gap                 => 5/100,
                :time_lim                => 60.0,
                :dont_scale_std          => false,
                :setzero_threshold       => 1e-3)

elseif case == 200
    # Define files
    pglib_case_name        = "pglib_opf_case200_activ_cost.m"
    powerflow_data_file    = data_folder*"case200_data.h5"
    powerflow_network_file = data_folder*"case200_network.h5"
    nn_parameter_file      = model_folder*"case200_neural_network.h5"
    BSON_file              = model_folder*"case200_BSON.bson"

    # Hyperparameters
    hyperparams = Dict(
                :num_ReLU                => 115,
                :dense_epoch_iterations  => 10000,
                :sparse_epoch_iterations => 10000,
                :adam                    => true,
                :dense_learning_rate     => 1e-3,
                :sparse_learning_rate    => 1e-3,
                :decay_tup               => (0.9,0.999),
                :print_val               => 200,
                :load_model              => false,
                :batch_size              => 100,
                :shuffle_data            => true,
                :percent_sparse          => 75/100,
                :mip_gap                 => 5/100,
                :time_lim                => 75.0,
                :dont_scale_std          => true,
                :setzero_threshold       => 2.5e-3)
end

# Call network data
network_data = PowerModels.parse_file("./pglib_opf/"*pglib_case_name)

# step 1: initial train
dense_model_data = train_powerflow_model(network_data, powerflow_data_file, powerflow_network_file, BSON_file, hyperparams)

# step 2: sparsify and re-train
sparse_model = train_powerflow_model_sparse(dense_model_data, hyperparams)

# %% step 3: save nn to BSON
#save_nn(sparse_model, dense_model_data, nn_parameter_file, BSON_file)

# vizualization
plot_nn_results(dense_model_data, sparse_model, 25)

# %% --------------------- step 4: tighten big-M and save big-M ---------------------
#ac_nn_model = run_big_M(network_data, powerflow_network_file, nn_parameter_file, hyperparams)
