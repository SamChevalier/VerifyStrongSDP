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

# Load data
data_folder  = "./data/"
model_folder = "./nn_models/"

# set seed and initialize plotly
Random.seed!(1);
plotly()

# test 1: underestimation of power injection at the bus with the largest limits
# test 2: overestimation of flow on the line with the largest limit
# test 3: build tables by looping over 5 buses
# test 4: MIQP vs STT -- V2

# %% test 1: TestPError =========================================
# ===============================================================
testname      = "TestPError"
testdata_file = data_folder*testname*".h5"
cases         = [14; 57; 118; 200]
masterlog_test1  = Dict(
    14  => Dict(),
    57  => Dict(),
    118 => Dict(),
    200 => Dict())

# Data Collection
for case in cases

    # Call case
    if case == 14
        pglib_case_name        = "pglib_opf_case14_ieee.m"
        powerflow_data_file    = data_folder*"case14_data.h5"
        powerflow_network_file = data_folder*"case14_network.h5"
        nn_parameter_file      = model_folder*"case14_neural_network.h5"
        num_runs               = 10
        num_new_constraints    = 100
    elseif case == 57
        pglib_case_name        = "pglib_opf_case57_ieee.m"
        powerflow_data_file    = data_folder*"case57_data.h5"
        powerflow_network_file = data_folder*"case57_network.h5"
        nn_parameter_file      = model_folder*"case57_neural_network.h5"
        num_runs               = 10
        num_new_constraints    = 200
    elseif case == 118
        pglib_case_name        = "pglib_opf_case118_ieee.m"
        powerflow_data_file    = data_folder*"case118_data.h5"
        powerflow_network_file = data_folder*"case118_network.h5"
        nn_parameter_file      = model_folder*"case118_neural_network.h5"
        num_runs               = 10
        num_new_constraints    = 250
    elseif case == 200
        pglib_case_name        = "pglib_opf_case200_activ_cost.m"
        powerflow_data_file    = data_folder*"case200_data.h5"
        powerflow_network_file = data_folder*"case200_network.h5"
        nn_parameter_file      = model_folder*"case200_neural_network.h5"
        num_runs               = 10
        num_new_constraints    = 350
    end

    # call the PowerModels network data
    network_data = PowerModels.parse_file("./pglib_opf/"*pglib_case_name)

    # build "ac_power_flow_model"
    ac_power_flow_model = build_acpf_model(network_data,powerflow_network_file)

    # build "nn_model" -- includes big-M
    ac_nn_model = build_nn_model(nn_parameter_file)

    # what bus shall we test? largest load
    pmax     = 0
    test_bus = 0
    for (ii,bus) in network_data["load"]
        pd = bus["pd"]
        if pd > pmax
            pmax = pd
            test_bus = bus["load_bus"]
        end
    end

    # Specify verification problem and index
    verification_params = Dict(
        :verification_routine      => "P", #"Ift2", #"Ift2", # "Q", # "P", "V2", "Ift2", "Itf2"
        :verification_index        => test_bus,
        :error_type                => "overestimation", #"underestimation", # "overestimation"
        :SDP_constraint_tol        => 1e-10,
        :sparse_SDP                => false,
        :initial_constraints       => [0],
        :active_constraints        => [0],
        :potential_new_constraints => [0],
        :voltage_constraints       => [0],
        :eta_constraints           => [0],
        :mosek_tols                => 1e-7,
        :num_new_constraints       => num_new_constraints,
        :iteration                 => 0,
        :solutions                 => Dict())

    # run stt
    sttlog = stt_algorithm(num_runs, ac_power_flow_model, ac_nn_model, verification_params)

    # run miqp
    timelim = 1.01*sttlog[:timelog][end]
    miqplog = miqp_verification(ac_power_flow_model, ac_nn_model, verification_params, timelim)

    masterlog_test1[case] = Dict(
        :sttlog  => sttlog,
        :miqplog => miqplog)
end

# write data to file
write_verificationdata(masterlog_test1,testdata_file,cases)

# %% test 2: TestI2Error =========================================
# ===============================================================
testname      = "TestI2Error"
testdata_file = data_folder*testname*".h5"
cases         = [14; 57; 118; 200]
masterlog_test2  = Dict(
    14  => Dict(),
    57  => Dict(),
    118 => Dict(),
    200 => Dict())

# Data Collection
for case in cases

    # Call case
    if case == 14
        pglib_case_name        = "pglib_opf_case14_ieee.m"
        powerflow_data_file    = data_folder*"case14_data.h5"
        powerflow_network_file = data_folder*"case14_network.h5"
        nn_parameter_file      = model_folder*"case14_neural_network.h5"
        num_runs               = 10
        num_new_constraints    = 100
    elseif case == 57
        pglib_case_name        = "pglib_opf_case57_ieee.m"
        powerflow_data_file    = data_folder*"case57_data.h5"
        powerflow_network_file = data_folder*"case57_network.h5"
        nn_parameter_file      = model_folder*"case57_neural_network.h5"
        num_runs               = 10
        num_new_constraints    = 200
    elseif case == 118
        pglib_case_name        = "pglib_opf_case118_ieee.m"
        powerflow_data_file    = data_folder*"case118_data.h5"
        powerflow_network_file = data_folder*"case118_network.h5"
        nn_parameter_file      = model_folder*"case118_neural_network.h5"
        num_runs               = 10
        num_new_constraints    = 250
    elseif case == 200
        pglib_case_name        = "pglib_opf_case200_activ_cost.m"
        powerflow_data_file    = data_folder*"case200_data.h5"
        powerflow_network_file = data_folder*"case200_network.h5"
        nn_parameter_file      = model_folder*"case200_neural_network.h5"
        num_runs               = 10
        num_new_constraints    = 350
    end

    # call the PowerModels network data
    network_data = PowerModels.parse_file("./pglib_opf/"*pglib_case_name)

    # build "ac_power_flow_model"
    ac_power_flow_model = build_acpf_model(network_data,powerflow_network_file)

    # build "nn_model" -- includes big-M
    ac_nn_model = build_nn_model(nn_parameter_file)

    # what line shall we test?
    test_line = argmax(abs.(ac_power_flow_model.Ift2_max))

    # Specify verification problem and index
    verification_params = Dict(
        :verification_routine      => "Ift2", #"Ift2", # "Q", # "P", "V2", "Ift2", "Itf2"
        :verification_index        => test_line,
        :error_type                => "underestimation", # "underestimation"
        :SDP_constraint_tol        => 1e-10,
        :sparse_SDP                => false,
        :initial_constraints       => [0],
        :active_constraints        => [0],
        :potential_new_constraints => [0],
        :voltage_constraints       => [0],
        :eta_constraints           => [0],
        :mosek_tols                => 1e-7,
        :num_new_constraints       => num_new_constraints,
        :iteration                 => 0,
        :solutions                 => Dict())

    # run stt
    sttlog = stt_algorithm(num_runs, ac_power_flow_model, ac_nn_model, verification_params)

    # run miqp
    timelim  = 1.01*sttlog[:timelog][end]
    miqplog = miqp_verification(ac_power_flow_model, ac_nn_model, verification_params, timelim)

    masterlog_test2[case] = Dict(
        :sttlog  => sttlog,
        :miqplog => miqplog)
end

# write data to file
write_verificationdata(masterlog_test2,testdata_file,cases)

# test 3: TestErrorTable ====================================
# ==============================================================
include("src/verification.jl")
cases = [14; 57; 118; 200]

masterlog_testc14t1  = Dict(1  => Dict(),2  => Dict(),3 => Dict(),4 => Dict(),5 => Dict())
masterlog_testc14t2  = Dict(1  => Dict(),2  => Dict(),3 => Dict(),4 => Dict(),5 => Dict())
masterlog_testc57t1  = Dict(1  => Dict(),2  => Dict(),3 => Dict(),4 => Dict(),5 => Dict())
masterlog_testc57t2  = Dict(1  => Dict(),2  => Dict(),3 => Dict(),4 => Dict(),5 => Dict())
masterlog_testc118t1 = Dict(1  => Dict(),2  => Dict(),3 => Dict(),4 => Dict(),5 => Dict())
masterlog_testc118t2 = Dict(1  => Dict(),2  => Dict(),3 => Dict(),4 => Dict(),5 => Dict())
masterlog_testc200t1 = Dict(1  => Dict(),2  => Dict(),3 => Dict(),4 => Dict(),5 => Dict())
masterlog_testc200t2 = Dict(1  => Dict(),2  => Dict(),3 => Dict(),4 => Dict(),5 => Dict())

# Data Collection
for case in cases

    # Call case
    if case == 14
        pglib_case_name        = "pglib_opf_case14_ieee.m"
        powerflow_data_file    = data_folder*"case14_data.h5"
        powerflow_network_file = data_folder*"case14_network.h5"
        nn_parameter_file      = model_folder*"case14_neural_network.h5"
        num_runs               = 10
        num_new_constraints    = 150
    elseif case == 57
        pglib_case_name        = "pglib_opf_case57_ieee.m"
        powerflow_data_file    = data_folder*"case57_data.h5"
        powerflow_network_file = data_folder*"case57_network.h5"
        nn_parameter_file      = model_folder*"case57_neural_network.h5"
        num_runs               = 10
        num_new_constraints    = 250
    elseif case == 118
        pglib_case_name        = "pglib_opf_case118_ieee.m"
        powerflow_data_file    = data_folder*"case118_data.h5"
        powerflow_network_file = data_folder*"case118_network.h5"
        nn_parameter_file      = model_folder*"case118_neural_network.h5"
        num_runs               = 10
        num_new_constraints    = 300
    elseif case == 200
        pglib_case_name        = "pglib_opf_case200_activ_cost.m"
        powerflow_data_file    = data_folder*"case200_data.h5"
        powerflow_network_file = data_folder*"case200_network.h5"
        nn_parameter_file      = model_folder*"case200_neural_network.h5"
        num_runs               = 10
        num_new_constraints    = 300
    end

    # call the PowerModels network data
    network_data = PowerModels.parse_file("./pglib_opf/"*pglib_case_name)

    # build "ac_power_flow_model"
    ac_power_flow_model = build_acpf_model(network_data,powerflow_network_file)

    # build "nn_model" -- includes big-M
    ac_nn_model = build_nn_model(nn_parameter_file)

    # choose buses with the most interconnections
    buses_set1  = reverse(sortperm(num_bus_connections(network_data)))
    buses_set2  = reverse(sortperm(num_bus_connections(network_data)))
    # buses_set1 = reverse(sortperm(abs.(ac_power_flow_model.Pinj_max - ac_power_flow_model.Pinj_min)))
    # buses_set2 = reverse(sortperm(abs.(ac_power_flow_model.Qinj_max - ac_power_flow_model.Qinj_min)))

    # what do we test for each case?
    for ii = 1:5
        if case == 14
            ind1 = buses_set1[ii]
            tst1 = "P"
            ind2 = buses_set2[ii]
            tst2 = "Q"
        elseif case == 57
            ind1 = buses_set1[ii]
            tst1 = "P"
            ind2 = buses_set2[ii]
            tst2 = "Q"
        elseif case == 118
            ind1 = buses_set1[ii]
            tst1 = "P"
            ind2 = buses_set2[ii]
            tst2 = "Q"
        elseif case == 200
            ind1 = buses_set1[ii]
            tst1 = "P"
            ind2 = buses_set2[ii]
            tst2 = "Q"
        end

        error_type1 = "underestimation"
        error_type2 = "underestimation"

        # ------- Test 1------------------------
        # Specify verification problem and index
        verification_params = Dict(
            :verification_routine      => tst1,
            :verification_index        => ind1,
            :error_type                => error_type1,
            :SDP_constraint_tol        => 0,
            :sparse_SDP                => false,
            :initial_constraints       => [0],
            :active_constraints        => [0],
            :potential_new_constraints => [0],
            :voltage_constraints       => [0],
            :eta_constraints           => [0],
            :mosek_tols                => 1e-6,
            :num_new_constraints       => num_new_constraints,
            :iteration                 => 0,
            :solutions                 => Dict())

        # run stt
        sttlog1 = stt_algorithm(num_runs, ac_power_flow_model, ac_nn_model, verification_params)

        # run miqp
        timelim  = sttlog1[:timelog][end]
        miqplog1 = miqp_verification(ac_power_flow_model, ac_nn_model, verification_params, timelim)

        if case == 14
            masterlog_testc14t1[ii] = Dict(
                :sttlog  => sttlog1,
                :miqplog => miqplog1)
        elseif case == 57
            masterlog_testc57t1[ii] = Dict(
                :sttlog  => sttlog1,
                :miqplog => miqplog1)
        elseif case == 118
            masterlog_testc118t1[ii] = Dict(
                :sttlog  => sttlog1,
                :miqplog => miqplog1)
        elseif case == 200
            masterlog_testc200t1[ii] = Dict(
                :sttlog  => sttlog1,
                :miqplog => miqplog1)
        end

        # ------- Test 2------------------------
        # Specify verification problem and index
        verification_params = Dict(
            :verification_routine      => tst2,
            :verification_index        => ind2,
            :error_type                => error_type2,
            :SDP_constraint_tol        => 1e-10,
            :sparse_SDP                => false,
            :initial_constraints       => [0],
            :active_constraints        => [0],
            :potential_new_constraints => [0],
            :voltage_constraints       => [0],
            :eta_constraints           => [0],
            :mosek_tols                => 1e-6,
            :num_new_constraints       => num_new_constraints,
            :iteration                 => 0,
            :solutions                 => Dict())

        # run stt
        sttlog2 = stt_algorithm(num_runs, ac_power_flow_model, ac_nn_model, verification_params)

        # run miqp
        timelim  = sttlog2[:timelog][end]
        miqplog2 = miqp_verification(ac_power_flow_model, ac_nn_model, verification_params, timelim)

        if case == 14
            masterlog_testc14t2[ii] = Dict(
                :sttlog  => sttlog2,
                :miqplog => miqplog2)
        elseif case == 57
            masterlog_testc57t2[ii] = Dict(
                :sttlog  => sttlog2,
                :miqplog => miqplog2)
        elseif case == 118
            masterlog_testc118t2[ii] = Dict(
                :sttlog  => sttlog2,
                :miqplog => miqplog2)
        elseif case == 200
            masterlog_testc200t2[ii] = Dict(
                :sttlog  => sttlog2,
                :miqplog => miqplog2)
        end
    end

    # Call case
    if case == 14
        testdata_file1 = data_folder*"testc14t1"*".h5"
        testdata_file2 = data_folder*"testc14t2"*".h5"
        write_verificationdata_table(masterlog_testc14t1,testdata_file1)
        write_verificationdata_table(masterlog_testc14t2,testdata_file2)
    elseif case == 57
        testdata_file1 = data_folder*"testc57t1"*".h5"
        testdata_file2 = data_folder*"testc57t2"*".h5"
        write_verificationdata_table(masterlog_testc57t1,testdata_file1)
        write_verificationdata_table(masterlog_testc57t2,testdata_file2)
    elseif case == 118
        testdata_file1 = data_folder*"testc118t1"*".h5"
        testdata_file2 = data_folder*"testc118t2"*".h5"
        write_verificationdata_table(masterlog_testc118t1,testdata_file1)
        write_verificationdata_table(masterlog_testc118t2,testdata_file2)
    elseif case == 200
        testdata_file1 = data_folder*"testc200t1"*".h5"
        testdata_file2 = data_folder*"testc200t2"*".h5"
        write_verificationdata_table(masterlog_testc200t1,testdata_file1)
        write_verificationdata_table(masterlog_testc200t2,testdata_file2)
    end
end

# %% Plot data
tau14t1  = zeros(5,1)
tau14t2  = zeros(5,1)
tau57t1  = zeros(5,1)
tau57t2  = zeros(5,1)
tau118t1 = zeros(5,1)
tau118t2 = zeros(5,1)
tau200t1 = zeros(5,1)
tau200t2 = zeros(5,1)
mrt14t1  = 0
mrt14t2  = 0
mrt57t1  = 0
mrt57t2  = 0
mrt118t1 = 0
mrt118t2 = 0
mrt200t1 = 0
mrt200t2 = 0

for ii = 1:5
    sbnd = masterlog_testc14t1[ii][:sttlog][:boundlog][end]
    mbnd = masterlog_testc14t1[ii][:miqplog][:boundlog][end]
    tau14t1[ii]  = mbnd/sbnd
    mrt14t1      = mrt14t1 + masterlog_testc14t1[ii][:sttlog][:timelog][end]

    sbnd = masterlog_testc14t2[ii][:sttlog][:boundlog][end]
    mbnd = masterlog_testc14t2[ii][:miqplog][:boundlog][end]
    tau14t2[ii]  = mbnd/sbnd
    mrt14t2      = mrt14t2 + masterlog_testc14t2[ii][:sttlog][:timelog][end]

    sbnd = masterlog_testc57t1[ii][:sttlog][:boundlog][end]
    mbnd = masterlog_testc57t1[ii][:miqplog][:boundlog][end]
    tau57t1[ii]  = mbnd/sbnd
    mrt57t1      = mrt57t1 + masterlog_testc57t1[ii][:sttlog][:timelog][end]

    sbnd = masterlog_testc57t2[ii][:sttlog][:boundlog][end]
    mbnd = masterlog_testc57t2[ii][:miqplog][:boundlog][end]
    tau57t2[ii]  = mbnd/sbnd
    mrt57t2      = mrt57t2 + masterlog_testc57t2[ii][:sttlog][:timelog][end]

    sbnd = masterlog_testc118t1[ii][:sttlog][:boundlog][end]
    mbnd = masterlog_testc118t1[ii][:miqplog][:boundlog][end]
    tau118t1[ii]  = mbnd/sbnd
    mrt118t1      = mrt118t1 + masterlog_testc118t1[ii][:sttlog][:timelog][end]

    sbnd = masterlog_testc118t2[ii][:sttlog][:boundlog][end]
    mbnd = masterlog_testc118t2[ii][:miqplog][:boundlog][end]
    tau118t2[ii]  = mbnd/sbnd
    mrt118t2      = mrt118t2 + masterlog_testc118t2[ii][:sttlog][:timelog][end]

    sbnd = masterlog_testc200t1[ii][:sttlog][:boundlog][end]
    mbnd = masterlog_testc200t1[ii][:miqplog][:boundlog][end]
    tau200t1[ii]  = mbnd/sbnd
    mrt200t1      = mrt200t1 + masterlog_testc200t1[ii][:sttlog][:timelog][end]

    sbnd = masterlog_testc200t2[ii][:sttlog][:boundlog][end]
    mbnd = masterlog_testc200t2[ii][:miqplog][:boundlog][end]
    tau200t2[ii]  = mbnd/sbnd
    mrt200t2      = mrt200t2 + masterlog_testc200t2[ii][:sttlog][:timelog][end]
end

mrt14t1  = mrt14t1/5
mrt14t2  = mrt14t2/5
mrt57t1  = mrt57t1/5
mrt57t2  = mrt57t2/5
mrt118t1 = mrt118t1/5
mrt118t2 = mrt118t2/5
mrt200t1 = mrt200t1/5
mrt200t2 = mrt200t2/5

# %% test 4: V2 Performance ====================================
# ==============================================================
# Call case
pglib_case_name        = "pglib_opf_case57_ieee.m"
powerflow_data_file    = data_folder*"case57_data.h5"
powerflow_network_file = data_folder*"case57_network.h5"
nn_parameter_file      = model_folder*"case57_neural_network.h5"
num_runs               = 20
num_new_constraints    = 450

# call the PowerModels network data
network_data = PowerModels.parse_file("./pglib_opf/"*pglib_case_name)

# build "ac_power_flow_model"
ac_power_flow_model = build_acpf_model(network_data,powerflow_network_file)

# build "nn_model" -- includes big-M
ac_nn_model = build_nn_model(nn_parameter_file)

# ------- Test 1------------------------
# Specify verification problem and index
verification_params = Dict(
    :verification_routine      => "V2",
    :verification_index        => 1,
    :error_type                => "overestimation",
    :SDP_constraint_tol        => 0,
    :sparse_SDP                => false,
    :initial_constraints       => [0],
    :active_constraints        => [0],
    :potential_new_constraints => [0],
    :voltage_constraints       => [0],
    :eta_constraints           => [0],
    :mosek_tols                => 1e-7,
    :num_new_constraints       => num_new_constraints,
    :iteration                 => 0,
    :solutions                 => Dict())

# run stt
sttlog1 = stt_algorithm(num_runs, ac_power_flow_model, ac_nn_model, verification_params)

# %% run miqp
timelim  = 1000.0
miqplog1 = miqp_verification(ac_power_flow_model, ac_nn_model, verification_params, timelim)

# %% Plot
c1 = 165/256
c2 = 42/256
c3 = 42/256
redd = RGB(c1,c2,c3)

c1 = 34/256  - 0.1
c2 = 139/256 - 0.1
c3 = 34/256  - 0.1
greenn = RGB(c1,c2,c3)

gr()
import Measures: Length, AbsoluteLength, Measure, BoundingBox, mm, cm, inch, pt, width, height, w, h
plot(miqplog1[:timelog], miqplog1[:boundlog], xaxis = :log, yaxis = :log, linewidth=4, opacity=.75, color = redd, label = "MIQP iteration", ylabel = "Error Bound (ẽ)", xlabel = "time (sec)")
plot!(sttlog1[:timelog], sttlog1[:boundlog], xaxis = :log, yaxis = :log, xlim = [0.08; 2000], color = 1,linewidth=1.5, label = "STT iteration")
plot!(sttlog1[:timelog], sttlog1[:boundlog], xaxis = :log, yaxis = :log, xlim = [0.08; 2000], seriestype = :scatter, color = :black, markersize=1.75, label = "",legend=:bottomleft,foreground_color_legend = nothing,legendfontsize=9,xtickfontsize = 9,ytickfontsize = 9, bottom_margin = 3Measures.mm, right_margin = 16Measures.mm, left_margin = 2.5Measures.mm)
plot!(twinx(),sttlog1[:timelog],sttlog1[:eig12log], xaxis = :log, yaxis = :log, xlim = [0.08; 2000], color = greenn, linestyle = :dashdot, linewidth=2.25, label = "STT eigenratio λ₁/λ₂",legend=:topright,foreground_color_legend = nothing,legendfontsize=9,xticks = [], ylabel = "Eigenvalue Ratio", yguidefont = font(10,greenn), yticks =[10^0.5; 10^1], ytickfontcolor = greenn,ytickfontsize = 9)
plot!(twinx(),sttlog1[:timelog],sttlog1[:eig12log], xaxis = :log, yaxis = :log, xlim = [0.08; 2000],seriestype = :scatter, color = :black, markersize=2.75, label = "",xticks = [], yguidefont = font(greenn), yticks =[])
p = plot!(size=(600,250))

#savefig("eigen_plot.pdf")

