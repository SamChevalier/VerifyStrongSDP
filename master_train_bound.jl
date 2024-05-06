# Activate the environment
using Pkg
Pkg.activate(".")

# %%
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

# Data Collection
for ii = 1:4
    if ii == 1
        case = 14
    elseif ii == 2
        case = 57
    elseif ii == 3
        case = 118
    elseif ii == 4
        case = 200
    end

    # go.
    #    
    # define the case
    if case == 14
        # Define files
        pglib_case_name        = "pglib_opf_case14_ieee.m"
        powerflow_data_file    = data_folder*"case14_data.h5"
        powerflow_network_file = data_folder*"case14_network.h5"
        load_scaling           = 50/100 # plus or minus
        num_pf_solves          = 250

    elseif case == 57
        # Define files
        pglib_case_name        = "pglib_opf_case57_ieee.m"
        powerflow_data_file    = data_folder*"case57_data.h5"
        powerflow_network_file = data_folder*"case57_network.h5"
        load_scaling           = 50/100 # plus or minus
        num_pf_solves          = 500
        
    elseif case == 118
        # Define files
        pglib_case_name        = "pglib_opf_case118_ieee.m"
        powerflow_data_file    = data_folder*"case118_data.h5"
        powerflow_network_file = data_folder*"case118_network.h5"
        load_scaling           = 50/100 # plus or minus
        num_pf_solves          = 750

    elseif case == 200
        # Define files
        pglib_case_name        = "pglib_opf_case200_activ_cost.m"
        powerflow_data_file    = data_folder*"case200_data.h5"
        powerflow_network_file = data_folder*"case200_network.h5"
        load_scaling           = 50/100 # plus or minus
        num_pf_solves          = 1000

    elseif case == 500
        # Define files
        pglib_case_name        = "pglib_opf_case500_goc.m"
        powerflow_data_file    = data_folder*"case500_data.h5"
        powerflow_network_file = data_folder*"case500_network.h5"
        load_scaling           = 50/100 # plus or minus
        num_pf_solves          = 3 #1000

    elseif case == 2000
        # Define files
        pglib_case_name        = "pglib_opf_case2000_goc.m"
        powerflow_data_file    = data_folder*"case2000_data.h5"
        powerflow_network_file = data_folder*"case2000_network.h5"
        load_scaling           = 50/100 # plus or minus
        num_pf_solves          = 3
    end

    # initialize powermodels
    PowerModels.silence()
    network_data = PowerModels.parse_file("./pglib_opf/"*pglib_case_name)

    # compute and apply current limits (network_data["branch"]["1"]["c_rating_a"]...)
    calc_current_limits!(network_data)

    # record system limits
    system_limits = get_system_limits(network_data,load_scaling)

    # update model: (1) remove apparent power flow limits and (2) transform loads into negative generators
    network_data_corrected = update_powermodel(network_data,load_scaling)

    # solve IPOPTs
    powerflow_data, Pgen_soln, Qgen_soln, Vm_soln = solve_IPOPTs(network_data_corrected,num_pf_solves)

    # save and record
    save_powerflow_data(system_limits,powerflow_data_file,powerflow_data,powerflow_network_file)
end

# %% Training
for ii = 1:4
    if ii == 1
        case = 14
    elseif ii == 2
        case = 57
    elseif ii == 3
        case = 118
    elseif ii == 4
        case = 200
    end

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
                    :num_ReLU                => 25,
                    :dense_epoch_iterations  => 40000,
                    :sparse_epoch_iterations => 40000,
                    :adam                    => true,
                    :dense_learning_rate     => 2e-4,
                    :sparse_learning_rate    => 2e-4,
                    :decay_tup               => (0.9,0.999),
                    :print_val               => 200,
                    :load_model              => false,
                    :batch_size              => 15,
                    :shuffle_data            => true,
                    :percent_sparse          => 50/100,
                    :mip_gap                 => 0.1/100,
                    :time_lim                => 60.0,
                    :dont_scale_std          => false,
                    :setzero_threshold       => 1e-3)
    elseif case == 57
        # Define files
        pglib_case_name        = "pglib_opf_case57_ieee.m"
        powerflow_data_file    = data_folder*"case57_data.h5"
        powerflow_network_file = data_folder*"case57_network.h5"
        nn_parameter_file      = model_folder*"case57_neural_network.h5"
        BSON_file              = model_folder*"case57_BSON.bson"

        # Hyperparameters
        hyperparams = Dict(
                    :num_ReLU                => 50,
                    :dense_epoch_iterations  => 40000,
                    :sparse_epoch_iterations => 40000,
                    :adam                    => true,
                    :dense_learning_rate     => 5e-4,
                    :sparse_learning_rate    => 5e-4,
                    :decay_tup               => (0.9,0.999),
                    :print_val               => 200,
                    :load_model              => false,
                    :batch_size              => 25,
                    :shuffle_data            => true,
                    :percent_sparse          => 50/100,
                    :mip_gap                 => 0.1/100,
                    :time_lim                => 60.0,
                    :dont_scale_std          => false,
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
                    :dense_epoch_iterations  => 25000,
                    :sparse_epoch_iterations => 25000,
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
                    :num_ReLU                => 100,
                    :dense_epoch_iterations  => 25000,
                    :sparse_epoch_iterations => 25000,
                    :adam                    => true,
                    :dense_learning_rate     => 1e-3,
                    :sparse_learning_rate    => 1e-3,
                    :decay_tup               => (0.9,0.999),
                    :print_val               => 200,
                    :load_model              => false,
                    :batch_size              => 75,
                    :shuffle_data            => true,
                    :percent_sparse          => 75/100,
                    :mip_gap                 => 5/100,
                    :time_lim                => 75.0,
                    :dont_scale_std          => false,
                    :setzero_threshold       => 1e-3)
    end

    # Call network data
    network_data = PowerModels.parse_file("./pglib_opf/"*pglib_case_name)

    # step 1: initial train
    dense_model_data = train_powerflow_model(network_data, powerflow_data_file, powerflow_network_file, BSON_file, hyperparams)

    # step 2: sparsify and re-train
    sparse_model = train_powerflow_model_sparse(dense_model_data, hyperparams)

    # step 3: save nn to BSON
    save_nn(sparse_model, dense_model_data, nn_parameter_file, BSON_file)

    # vizualization
    # plot_nn_results(dense_model_data, sparse_model, 100)

    # --------------------- step 4: tighten big-M and save big-M ---------------------
    ac_nn_model = run_big_M(network_data, powerflow_network_file, nn_parameter_file, hyperparams)

end