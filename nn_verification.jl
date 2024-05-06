# Activate the environment
using Pkg
Pkg.activate(".")

# Load the packages
using JuMP
using HDF5
using Plots
using Mosek
using Gurobi
using Graphs
using Random
using Base.Sort
using MosekTools
using PowerModels
using SparseArrays
using SolverLogging
using LinearAlgebra
using InvertedIndices
using MathOptInterface
using BSON: @save, @load

MOI = MathOptInterface

# Add function files
include("src/power_system_utilities.jl")
include("src/model_structures.jl")
include("src/data_collection.jl")
include("src/verification.jl")
include("src/training.jl")

# set seed and initialize plotly
Random.seed!(1);
plotly()

# Define case
case = 14

# Load data
data_folder  = "./data/"
model_folder = "./nn_models/"

# Call case
if case == 14
    pglib_case_name        = "pglib_opf_case14_ieee.m"
    powerflow_data_file    = data_folder*"case14_data.h5"
    powerflow_network_file = data_folder*"case14_network.h5"
    nn_parameter_file      = model_folder*"case14_neural_network.h5"
elseif case == 57
    pglib_case_name        = "pglib_opf_case57_ieee.m"
    powerflow_data_file    = data_folder*"case57_data.h5"
    powerflow_network_file = data_folder*"case57_network.h5"
    nn_parameter_file      = model_folder*"case57_neural_network.h5"
elseif case == 118
    pglib_case_name        = "pglib_opf_case118_ieee.m"
    powerflow_data_file    = data_folder*"case118_data.h5"
    powerflow_network_file = data_folder*"case118_network.h5"
    nn_parameter_file      = model_folder*"case118_neural_network.h5"
elseif case == 200
    pglib_case_name        = "pglib_opf_case200_activ_cost.m"
    powerflow_data_file    = data_folder*"case200_data.h5"
    powerflow_network_file = data_folder*"case200_network.h5"
    nn_parameter_file      = model_folder*"case200_neural_network.h5"
end

# call the PowerModels network data
network_data = PowerModels.parse_file("./pglib_opf/"*pglib_case_name)

# build "ac_power_flow_model"
ac_power_flow_model = build_acpf_model(network_data,powerflow_network_file)

# build "nn_model" -- includes big-M
ac_nn_model = build_nn_model(nn_parameter_file)

################################################
################################################
# %% Specify verification problem and index
verification_params = Dict(
    :verification_routine      => "P",#"Ift2", #"Ift2", # "Q", # "P", "V2", "Ift2", "Itf2"
    :verification_index        => 5,
    :error_type                => "underestimation",# "overestimation", #"underestimation"
    :SDP_constraint_tol        => 1e-10,
    :sparse_SDP                => false,
    :initial_constraints       => [0],
    :active_constraints        => [0],
    :potential_new_constraints => [0],
    :voltage_constraints       => [0],
    :eta_constraints           => [0],
    :mosek_tols                => 1e-6,
    :num_new_constraints       => 100,
    :iteration                 => 1,
    :solutions                 => Dict())

# %% initial verification
Results, verification_model, verification_params = strong_SDP_verification(ac_power_flow_model, ac_nn_model, verification_params);

# %% loop verification
Results, verification_model, verification_params = iterate_strong_SDP_verification(verification_model, verification_params);

# %% -- MIQP
timelim = 1000.0
miqplog,vv,xx = miqp_verification(ac_power_flow_model, ac_nn_model, verification_params, timelim)

# %% plot
timevec, boundvec, eig12vec, eig13vec = sdp_data(verification_params)
plot(miqplog[:timelog],miqplog[:boundlog])
plot!(timevec, boundvec, yaxis = :log, xaxis = :log)