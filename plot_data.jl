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

pglib_case_name        = "pglib_opf_case57_ieee.m"
    powerflow_data_file    = data_folder*"case57_data.h5"
    powerflow_network_file = data_folder*"case57_network.h5"
# Load Data

c = h5open(powerflow_data_file, "r") do file
    # Training Data (70%)
    global Vm_TrDat    = read(file, "Vm_TrainData")
    global Theta_TrDat = read(file, "Theta_TrainData")
    global Vr_TrDat    = read(file, "Vr_TrainData")
    global Vi_TrDat    = read(file, "Vi_TrainData")
    global Pinj_TrDat  = read(file, "Pinj_TrainData")
    global Qinj_TrDat  = read(file, "Qinj_TrainData")
    global Sft_TrDat   = read(file, "Sft_TrainData")
    global Stf_TrDat   = read(file, "Stf_TrainData")
    global Ift2_TrDat  = read(file, "Ift2_TrainData")
    global Itf2_TrDat  = read(file, "Itf2_TrainData")

    # Validation Data (15%)
    global Vm_ValDat    = read(file, "Vm_ValidationData")
    global Theta_ValDat = read(file, "Theta_ValidationData")
    global Vr_ValDat    = read(file, "Vr_ValidationData")
    global Vi_ValDat    = read(file, "Vi_ValidationData")
    global Pinj_ValDat  = read(file, "Pinj_ValidationData")
    global Qinj_ValDat  = read(file, "Qinj_ValidationData")
    global Sft_ValDat   = read(file, "Sft_ValidationData")
    global Stf_ValDat   = read(file, "Stf_ValidationData")
    global Ift2_ValDat  = read(file, "Ift2_ValidationData")
    global Itf2_ValDat  = read(file, "Itf2_ValidationData")

    # test
    global Vm_TestDat    = read(file, "Vm_TestData")

end

# %% Scatter Plots
#using LaTeXStrings
using Plots.PlotMeasures
gr()

#plot([Vm_TrDat[1:5,:] Vm_ValDat[1:5,:]  Vm_TestDat[1:5,:] ], color = :black, alpha = 0.1, linewidth=1.5, ylabel="Voltage (p.u.)", xlabel="Bus Number")

# %%
plot([Vm_TrDat Vm_ValDat  Vm_TestDat], color = :steelblue, alpha = 0.1, linewidth=1.5, ylabel="Voltage Magnitude (p.u.)", xlabel="Bus Number", legend = false)
plot!(0.94*ones(57,1), linestyle = :dash, color = 2, linewidth=1.5)
plot!(1.06*ones(57,1), linestyle = :dash, color = 2, linewidth=1.5)
p = plot!(size=(600,300))

# Plots.savefig("voltage_profile.pdf")

# %% ------------------- Set skip_for_plotting = 1!!!!!!!!!!!!!!
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
using LinearAlgebra
using InvertedIndices
using BSON: @save, @load

# Add function files
include("src/power_system_utilities.jl")
include("src/model_structures.jl")
include("src/data_collection.jl")
include("src/verification.jl")
include("src/training.jl")

# set seed and initialize plotly
Random.seed!(1);
plotly()

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

# assess constraints for feasible dual (bounded primal)
Results, verification_model, verification_params = strong_SDP_verification(ac_power_flow_model, ac_nn_model, verification_params);
om1 = Results[:sorted_Omega]
er1 = verification_params[:solutions][1][:eig_ratio_12]

Results, verification_model, verification_params = iterate_strong_SDP_verification(verification_model, verification_params);
om2 = Results[:sorted_Omega]
er2 = verification_params[:solutions][2][:eig_ratio_12]

Results, verification_model, verification_params = iterate_strong_SDP_verification(verification_model, verification_params);
om3 = Results[:sorted_Omega]
er3 = verification_params[:solutions][3][:eig_ratio_12]

Results, verification_model, verification_params = iterate_strong_SDP_verification(verification_model, verification_params);
om4 = Results[:sorted_Omega]
er4 = verification_params[:solutions][4][:eig_ratio_12]

Results, verification_model, verification_params = iterate_strong_SDP_verification(verification_model, verification_params);
om5 = Results[:sorted_Omega]
er5 = verification_params[:solutions][5][:eig_ratio_12]

# %%
#using LaTeXStrings
#using GR
gr()

om1_alt = [om1[1:300]]#; om1[end-1000:end]]
om2_alt = [om2[1:300]]#; om2[end-1000:end]]
om3_alt = [om3[1:300]]#; om3[end-1000:end]]
om4_alt = [om4[1:300]]#; om4[end-1000:end]]
om5_alt = [om5[1:300]]#; om5[end-1000:end]]

plot(om1_alt,linewidth=1.75,label = "STT iteration 0",xlabel = "(Sorted) Constraint Number Ωᵢⱼ",ylabel = "Constraint Violation")
plot!(om2_alt,linewidth=1.75,label = "STT iteration 1")
plot!(om3_alt,linewidth=1.75,label = "STT iteration 2")
plot!(om4_alt,linewidth=1.75,label = "STT iteration 3")
plot!(om5_alt,linewidth=3,legend=:bottomright,foreground_color_legend = nothing,label = "STT iteration 4", yticks = [-3e4; -2e4; -1e4; 0], xticks = [0; 8; 50; 100; 150; 200], linestyle = :dashdot,legendfontsize=10, xlim = [0; 200])
# xaxis=:log  Ωᵢⱼ , fontfamily="JuliaMono"
# ,xticks = [0; 25; 50; 100; 150; 200; 250]

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
#plot!(rectangle(25,5.2e4,0,-5.2e4), opacity=.25, color=:black,label=false)
plot!(rectangle(8,3.25e4,0,-3.25e4), opacity=.25, color=:black,label=false)


annotate!([(54, -0.72e4, ("constraints to enforce", 8))])
annotate!([(54, -0.88e4, ("at the next iteration", 8))])
plot!([25,10],[-0.8e4,-0.8e4],arrow=true,color=:black,linewidth=1,label="")
#annotate!(50, -3.5e4, "the next iteration")

# Plots.savefig("constraint_violation.pdf")

p = plot!(size=(600,300))




# %% NN results
pglib_case_name        = "pglib_opf_case14_ieee.m"
powerflow_data_file    = data_folder*"case14_data.h5"
powerflow_network_file = data_folder*"case14_network.h5"
nn_parameter_file      = model_folder*"case14_neural_network.h5"
ac_nn_model            = build_nn_model(nn_parameter_file)

#powerflow_data_file    = data_folder*"case118_data.h5"
#nn_parameter_file      = model_folder*"case118_neural_network.h5"
#ac_nn_model            = build_nn_model(nn_parameter_file)
#powerflow_network_file = data_folder*"case118_network.h5"

# Load newtork information
c = h5open(powerflow_network_file, "r") do file
    global ref_bus      = read(file, "ref_bus") 
    global num_buses    = read(file, "num_buses") 
    global num_lines    = read(file, "num_lines") 
end

c = h5open(powerflow_data_file, "r") do file
        # Training Data (70%)
        global Vm_TrDat    = read(file, "Vm_TrainData")
        global Theta_TrDat = read(file, "Theta_TrainData")
        global Vr_TrDat    = read(file, "Vr_TrainData")
        global Vi_TrDat    = read(file, "Vi_TrainData")
        global Pinj_TrDat  = read(file, "Pinj_TrainData")
        global Qinj_TrDat  = read(file, "Qinj_TrainData")
        global Sft_TrDat   = read(file, "Sft_TrainData")
        global Stf_TrDat   = read(file, "Stf_TrainData")
        global Ift2_TrDat  = read(file, "Ift2_TrainData")
        global Itf2_TrDat  = read(file, "Itf2_TrainData")

        # Validation Data (15%)
        global Vm_ValDat    = read(file, "Vm_ValidationData")
        global Theta_ValDat = read(file, "Theta_ValidationData")
        global Vr_ValDat    = read(file, "Vr_ValidationData")
        global Vi_ValDat    = read(file, "Vi_ValidationData")
        global Pinj_ValDat  = read(file, "Pinj_ValidationData")
        global Qinj_ValDat  = read(file, "Qinj_ValidationData")
        global Sft_ValDat   = read(file, "Sft_ValidationData")
        global Stf_ValDat   = read(file, "Stf_ValidationData")
        global Ift2_ValDat  = read(file, "Ift2_ValidationData")
        global Itf2_ValDat  = read(file, "Itf2_ValidationData")

        global Vm_TestData    = read(file, "Vm_TestData")
        global Theta_TestDat = read(file, "Theta_TestData")
        global Vr_TestData    = read(file, "Vr_TestData")
        global Vi_TestData    = read(file, "Vi_TestData")
        global Pinj_TestData  = read(file, "Pinj_TestData")
        global Qinj_TestData  = read(file, "Qinj_TestData")

        global Ift2_TestData  = read(file, "Ift2_TestData")
        global Itf2_TestData  = read(file, "Itf2_TestData")
end

W0       = sparse(ac_nn_model.W0)
b0       = ac_nn_model.b0
W1       = sparse(ac_nn_model.W1)
b1       = ac_nn_model.b1

In_Mean  = ac_nn_model.In_Mean
In_STD   = ac_nn_model.In_STD
Out_Mean = ac_nn_model.Out_Mean
Out_STD  = ac_nn_model.Out_STD
J0       = ac_nn_model.J0
r0       = ac_nn_model.r0

ind              = 20
nn_input         = [Vr_TestData[:,ind]; Vi_TestData[Not(ref_bus),ind]]
nn_input_normal  = (nn_input - In_Mean)./In_STD
x_ReLU_in        = W0*nn_input_normal + b0
x_ReLU_out       = max.(0,x_ReLU_in)
nn_output_normal = W1*x_ReLU_out + b1
nn_output_raw    = Out_STD.*nn_output_normal + Out_Mean
predicted_output = nn_output_raw + J0*nn_input + r0
output           = [Vm_TestData.^2; Pinj_TestData; Qinj_TestData; Ift2_TestData; Itf2_TestData]

# subplots
c1 = 165/256
c2 = 42/256
c3 = 42/256
redd = RGB(c1,c2,c3)

sample = ind

l  = @layout [a b; c d]
p1 = plot(predicted_output[1:num_buses],legend = false,linewidth=4, opacity=.75, color = redd, linestyle = :dashdot) # Plot V^2
p1 = plot!(output[1:num_buses,sample], legend = false,ylabel = "Sqr. Voltage", xticks = [1; 5; 10; 14],xlabel = "Bus Number", color = 1,linewidth=1.5)

p2 = plot(predicted_output[num_buses+1:2*num_buses],legend = false, ylabel = "P Injection", xticks = [1; 5; 10; 14],xlabel = "Bus Number",linewidth=4, opacity=.75, color = redd, linestyle = :dashdot) # Plot P
p2 = plot!(output[num_buses+1:2*num_buses,sample], legend = false,color = 1,linewidth=1.5)
p3 = plot(predicted_output[2*num_buses+1:3*num_buses],legend = false, ylabel = "Q Injection", xticks = [1; 5; 10; 14],xlabel = "Bus Number",linewidth=4, opacity=.75, color = redd, linestyle = :dashdot, yticks = [-0.3; 0; 0.3]) # Plot Q
p3 = plot!(output[2*num_buses+1:3*num_buses,sample], legend = false,color = 1,linewidth=1.5)
p4 = plot(predicted_output[3*num_buses+1:(3*num_buses+num_lines)], ylabel = "Sqr. Current", xticks = [1; 5; 10; 15; 20],xlabel = "Line Number",linewidth=4, opacity=.75, color = redd, linestyle = :dashdot,label = "NN Prediction",foreground_color_legend = nothing) # Plot I^2
p4 = plot!(output[3*num_buses+1:(3*num_buses+num_lines),sample],color = 1, label = "Ground Truth",linewidth=1.5)
plot(p1, p2, p3, p4, layout = l)

p = plot!(size=(600,320))

# Plots.savefig("nn_prediction.pdf")

# %% call the PowerModels network data
network_data = PowerModels.parse_file("./pglib_opf/"*pglib_case_name)

# build "ac_power_flow_model"
ac_power_flow_model = build_acpf_model(network_data,powerflow_network_file)

# %% Plot -- Test 1
testname      = "TestPError"
testdata_file = data_folder*testname*".h5"
cases         = [14; 57; 118; 200]

results_test1 = Dict(
    14  => Dict(),
    57  => Dict(),
    118 => Dict(),
    200 => Dict())

# Load Data
for case in cases
    casestr = string(case)
    c = h5open(testdata_file, "r") do file
        results_test1[case] = Dict(
            :sttlog_timelog   => read(file, "sttlog_timelog_"*casestr),
            :sttlog_boundlog  => read(file, "sttlog_boundlog_"*casestr),
            :sttlog_eig12log  => read(file, "sttlog_eig12log_"*casestr),
            :sttlog_eig13log  => read(file, "sttlog_eig13log_"*casestr),
            :miqplog_timelog  => read(file, "miqplog_timelog_"*casestr),
            :miqplog_boundlog => read(file, "miqplog_boundlog_"*casestr),
            :miqplog_bestlog  => read(file, "miqplog_bestlog_"*casestr),
            :miqplog_gaplog   => read(file, "miqplog_gaplog_"*casestr))
    end
end

# Plot
gr()

l  = @layout [a b; c d]
# subplots
c1 = 165/256
c2 = 42/256
c3 = 42/256
redd = RGB(c1,c2,c3)

n1 = length(results_test1[14][:miqplog_timelog])
n2 = length(results_test1[57][:miqplog_timelog])
n3 = length(results_test1[118][:miqplog_timelog])
n4 = length(results_test1[200][:miqplog_timelog])

pltidx1 = [collect(1:100); collect(101:10:n1)]
pltidx2 = [collect(1:100); collect(101:10:n2)]
pltidx3 = [collect(1:100); collect(101:10:n3)]
pltidx4 = [collect(1:100); collect(101:10:n4)]

case = 14
# layout = Layout(legend=attr(orientation="h"))

p1 = plot(results_test1[case][:miqplog_timelog][pltidx1], results_test1[case][:miqplog_boundlog][pltidx1], yaxis = :log, xaxis = :log10, linewidth=4, opacity=.75, color = redd, label = "MIQP iteration", foreground_color_legend = nothing) #legend = (position = :bottom, titleposition = :left)) #foreground_color_legend = nothing)
p1 = plot!(results_test1[case][:sttlog_timelog], results_test1[case][:sttlog_boundlog], ylabel = "Error Bound (ẽ)", xlabel = "", color = 1,linewidth=1.5, label = "STT iteration", xlim = [0.007; 15], xticks = [10^(-2); 10^(-1); 10^0; 10^1])
p1 = plot!(results_test1[case][:sttlog_timelog], results_test1[case][:sttlog_boundlog], seriestype = :scatter, ylabel = "Error Bound (ẽ)", xlabel = "", color = :black, markersize=1.75, label = "")

annotate!([(0.025, 0.14, ("(14 bus)", "Helvetica Bold", 10))])

case = 57
p2 = plot(results_test1[case][:miqplog_timelog][pltidx2], results_test1[case][:miqplog_boundlog][pltidx2], yaxis = :log, xaxis = :log, linewidth=4, opacity=.75, color = redd, legend = false)
p2 = plot!(results_test1[case][:sttlog_timelog], results_test1[case][:sttlog_boundlog], ylabel = "", xlabel = "", color = 1,linewidth=1.5, legend = false, xticks = [ 10^(-1); 10^0; 10^1; 10^2])
p2 = plot!(results_test1[case][:sttlog_timelog], results_test1[case][:sttlog_boundlog], seriestype = :scatter, ylabel = "", xlabel = "", color = :black, markersize=1.75, label = "")
annotate!([(0.2, 5, ("(57 bus)", "Helvetica Bold", 10))])

case = 118
p3 = plot(results_test1[case][:miqplog_timelog][pltidx3], results_test1[case][:miqplog_boundlog][pltidx3], yaxis = :log, xaxis = :log, linewidth=4, opacity=.75, color = redd, legend = false)#, yticks = [10; 100])#, ylim = [10; 100])
p3 = plot!(results_test1[case][:sttlog_timelog], results_test1[case][:sttlog_boundlog], ylabel = "Error Bound (ẽ)", xlabel = "time (sec)", color = 1,linewidth=1.5, legend = false, xticks = [ 10^(-1); 10^0; 10^1; 10^2])
p3 = plot!(results_test1[case][:sttlog_timelog], results_test1[case][:sttlog_boundlog], seriestype = :scatter, ylabel = "Error Bound (ẽ)", xlabel = "time (sec)", color = :black, markersize=1.75, label = "")
annotate!([(0.4, 19, ("(118 bus)", "Helvetica Bold", 10))])


case = 200
p4 = plot(results_test1[case][:miqplog_timelog][pltidx4], results_test1[case][:miqplog_boundlog][pltidx4], yaxis = :log, xaxis = :log, linewidth=4, opacity=.75, color = redd, legend = false)
p4 = plot!(results_test1[case][:sttlog_timelog], results_test1[case][:sttlog_boundlog], ylabel = "", xlabel = "time (sec)", color = 1,linewidth=1.5, legend = false, xticks = [ 10^(0); 10^1; 10^2; 10^3])
p4 = plot!(results_test1[case][:sttlog_timelog], results_test1[case][:sttlog_boundlog], seriestype = :scatter, ylabel = "", xlabel = "time (sec)", color = :black, markersize=1.75, label = "", xlim = [0.15; 1300])
annotate!([(0.75, 4, ("(200 bus)", "Helvetica Bold", 10))])

plot(p1, p2, p3, p4, layout = l)
p = plot!(size=(600,350))
# savefig("power_bound_race.pdf")
    

# %% Plot -- Test 2
testname      = "TestI2Error"
testdata_file = data_folder*testname*".h5"
cases         = [14; 57; 118; 200]

results_test2 = Dict(
    14  => Dict(),
    57  => Dict(),
    118 => Dict(),
    200 => Dict())

# Load Data
for case in cases
    casestr = string(case)
    c = h5open(testdata_file, "r") do file
        results_test2[case] = Dict(
            :sttlog_timelog   => read(file, "sttlog_timelog_"*casestr),
            :sttlog_boundlog  => read(file, "sttlog_boundlog_"*casestr),
            :sttlog_eig12log  => read(file, "sttlog_eig12log_"*casestr),
            :sttlog_eig13log  => read(file, "sttlog_eig13log_"*casestr),
            :miqplog_timelog  => read(file, "miqplog_timelog_"*casestr),
            :miqplog_boundlog => read(file, "miqplog_boundlog_"*casestr),
            :miqplog_bestlog  => read(file, "miqplog_bestlog_"*casestr),
            :miqplog_gaplog   => read(file, "miqplog_gaplog_"*casestr))
    end
end

# Plot
gr()

l  = @layout [a b; c d]
# subplots
c1 = 165/256
c2 = 42/256
c3 = 42/256
redd = RGB(c1,c2,c3)

n1 = length(results_test2[14][:miqplog_timelog])
n2 = length(results_test2[57][:miqplog_timelog])
n3 = length(results_test2[118][:miqplog_timelog])
n4 = length(results_test2[200][:miqplog_timelog])

pltidx1 = [collect(1:100); collect(101:10:n1)]
pltidx2 = [collect(1:100); collect(101:10:n2)]
pltidx3 = [collect(1:100); collect(101:10:n3)]
pltidx4 = [collect(1:100); collect(101:10:n4)]

case = 14
# layout = Layout(legend=attr(orientation="h"))

p1 = plot(results_test2[case][:miqplog_timelog][pltidx1], results_test2[case][:miqplog_boundlog][pltidx1], yaxis = :log, xaxis = :log10, linewidth=4, opacity=.75, color = redd, label = "MIQP iteration", foreground_color_legend = nothing,legend=:bottomleft) #legend = (position = :bottom, titleposition = :left)) #foreground_color_legend = nothing)
p1 = plot!(results_test2[case][:sttlog_timelog], results_test2[case][:sttlog_boundlog], ylabel = "Error Bound (ẽ)", xlabel = "", color = 1,linewidth=1.5, label = "STT iteration", xticks = [10^(-2); 10^(-1); 10^0; 10^1])
p1 = plot!(results_test2[case][:sttlog_timelog], results_test2[case][:sttlog_boundlog], seriestype = :scatter, ylabel = "Error Bound (ẽ)", xlabel = "", color = :black, markersize=1.75, label = "")

annotate!([(0.03, 7.24, ("(14 bus)", "Helvetica Bold", 10))])

case = 57
p2 = plot(results_test2[case][:miqplog_timelog][pltidx2], results_test2[case][:miqplog_boundlog][pltidx2], yaxis = :log, xaxis = :log, linewidth=4, opacity=.75, color = redd, legend = false)
p2 = plot!(results_test2[case][:sttlog_timelog], results_test2[case][:sttlog_boundlog], ylabel = "", xlabel = "", color = 1,linewidth=1.5, legend = false, xticks = [ 10^(-1); 10^0; 10^1; 10^2])
p2 = plot!(results_test2[case][:sttlog_timelog], results_test2[case][:sttlog_boundlog], seriestype = :scatter, ylabel = "", xlabel = "", color = :black, markersize=1.75, label = "")
annotate!([(0.13, 4.9, ("(57 bus)", "Helvetica Bold", 10))])

case = 118
p3 = plot(results_test2[case][:miqplog_timelog][pltidx3], results_test2[case][:miqplog_boundlog][pltidx3], yaxis = :log, xaxis = :log, linewidth=4, opacity=.75, color = redd, legend = false)#, yticks = [10; 100])#, ylim = [10; 100])
p3 = plot!(results_test2[case][:sttlog_timelog], results_test2[case][:sttlog_boundlog], ylabel = "Error Bound (ẽ)", xlabel = "time (sec)", color = 1,linewidth=1.5, legend = false, xticks = [ 10^(-1); 10^0; 10^1; 10^2])
p3 = plot!(results_test2[case][:sttlog_timelog], results_test2[case][:sttlog_boundlog], seriestype = :scatter, ylabel = "Error Bound (ẽ)", xlabel = "time (sec)", color = :black, markersize=1.75, label = "")
annotate!([(0.4, 500, ("(118 bus)", "Helvetica Bold", 10))])

case = 200
p4 = plot(results_test2[case][:miqplog_timelog][pltidx4], results_test2[case][:miqplog_boundlog][pltidx4], yaxis = :log, xaxis = :log, linewidth=4, opacity=.75, color = redd, legend = false)
p4 = plot!(results_test2[case][:sttlog_timelog], results_test2[case][:sttlog_boundlog], ylabel = "", xlabel = "time (sec)", color = 1,linewidth=1.5, legend = false, xticks = [ 10^(0); 10^1; 10^2; 10^3])
p4 = plot!(results_test2[case][:sttlog_timelog], results_test2[case][:sttlog_boundlog], seriestype = :scatter, ylabel = "", xlabel = "time (sec)", color = :black, markersize=1.75, label = "")
annotate!([(0.9, 120, ("(200 bus)", "Helvetica Bold", 10))])

plot(p1, p2, p3, p4, layout = l)
p = plot!(size=(600,350))
# savefig("current_bound_race.pdf")

