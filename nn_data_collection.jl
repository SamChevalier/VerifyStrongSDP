# Activate the environment
using Pkg
Pkg.activate(".")

# %%
using JuMP
using HDF5
using Plots
using Ipopt
using Random
using PowerModels
using SparseArrays
using LinearAlgebra
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
case = 14

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
