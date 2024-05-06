# ------------------- Set skip_for_plotting = 1!!!!!!!!!!!!!!
using Pkg
Pkg.activate(".")

# %% Load the packages
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
using LinearAlgebra
using InvertedIndices
using BSON: @save, @load

# Add function files
include("../src/power_system_utilities.jl")
include("../src/model_structures.jl")
include("../src/data_collection.jl")
include("../src/verification.jl")
include("../src/training.jl")

# set seed and initialize plotly
Random.seed!(1);
plotly()
gr()

# %% %%%%%%%%%%%%%%% Demo 1: constraint violation %%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Load data
data_folder  = "./data/"
model_folder = "./nn_models/"

pglib_case_name        = "pglib_opf_case14_ieee.m"
powerflow_data_file    = data_folder*"case14_data.h5"
powerflow_network_file = data_folder*"case14_network.h5"
nn_parameter_file      = model_folder*"case14_neural_network.h5"

# call the PowerModels network data
network_data = PowerModels.parse_file("./pglib_opf/"*pglib_case_name)

# build "ac_power_flow_model"
ac_power_flow_model = build_acpf_model(network_data,powerflow_network_file)

# build "nn_model" -- includes big-M
ac_nn_model = build_nn_model(nn_parameter_file)

################################################
################################################
# Specify verification problem and index
verification_params = Dict(
    :verification_routine      => "Ift2",
    :verification_index        => 1,
    :error_type                => "overestimation", # "overestimation",
    :SDP_constraint_tol        => 1e-10,
    :sparse_SDP                => false,
    :initial_constraints       => [0],
    :active_constraints        => [0],
    :potential_new_constraints => [0],
    :voltage_constraints       => [0],
    :eta_constraints           => [0],
    :mosek_tols                => 1e-6,
    :num_new_constraints       => 8,
    :iteration                 => 1,
    :solutions                 => Dict())
ncs = verification_params[:num_new_constraints]

# assess constraints for feasible dual (bounded primal)
verification_params[:skip_for_plotting]          = 1
Results, verification_model, verification_params = strong_SDP_verification(ac_power_flow_model, ac_nn_model, verification_params);
om1 = Results[:sorted_Omega]
er1 = verification_params[:solutions][1][:eig_ratio_12]

# round 0: plot all constraints
plot(om1, linewidth=1.75,label = "No constraints enforced",xlabel = "Sorted Constraint Number (135,424 total)",ylabel = "Constraint Violation (negative = bad)",legend=:bottomright)



# %% round 1: plot worst offenders
om1_alt = [om1[1:200]]
plot(om1_alt, linewidth=1.75,label = "No constraints enforced",xlabel = "Sorted Constraint Number (200/135,424)",ylabel = "Constraint Violation",legend=:bottomright,foreground_color_legend = nothing, yticks = [-3e4; -2e4; -1e4; 0], xticks = [0; 8; 50; 100; 150; 200], legendfontsize=10, xlim = [0; 200])

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
plot!(rectangle(8,3.25e4,0,-3.25e4), opacity=.25, color=:black,label=false)
annotate!([(54, -0.72e4, ("constraints to enforce", 8))])
annotate!([(54, -0.88e4, ("at the next iteration", 8))])
plot!([25,10],[-0.8e4,-0.8e4],arrow=true,color=:black,linewidth=1,label="")
p = plot!(size=(600,300))

# %% round 2: plot worst offenders
Results, verification_model, verification_params = iterate_strong_SDP_verification(verification_model, verification_params);
om2 = Results[:sorted_Omega]
om2_alt = [om2[1:200]]
plot!(om2_alt, linewidth=1.75,label = "$ncs constraints enforced")

# %% round 3: plot worst offenders
Results, verification_model, verification_params = iterate_strong_SDP_verification(verification_model, verification_params);
om3 = Results[:sorted_Omega]
om3_alt = [om3[1:200]]
plot!(om3_alt, linewidth=1.75,label = "$(2*ncs) constraints enforced")

# %% round 4: plot worst offenders
Results, verification_model, verification_params = iterate_strong_SDP_verification(verification_model, verification_params);
om4 = Results[:sorted_Omega]
om4_alt = [om4[1:200]]
plot!(om4_alt, linewidth=1.75,label = "$(3*ncs) constraints enforced")

# %% round 4: plot worst offenders
Results, verification_model, verification_params = iterate_strong_SDP_verification(verification_model, verification_params);
om5 = Results[:sorted_Omega]
om5_alt = [om5[1:200]]
plot!(om5_alt, linewidth=1.75,label = "$(4*ncs) constraints enforced")