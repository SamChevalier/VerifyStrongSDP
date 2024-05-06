# Save training, validation, and testing data and model information to file
function save_powerflow_data(system_limits::Dict,powerflow_data_file::String,powerflow_data::Dict,powerflow_network_file::String)

    # Convert data to Matrix{Float64}
    Vm_Data    = convert(Matrix{Float64}, powerflow_data[:Vm_Data])
    Theta_Data = convert(Matrix{Float64}, powerflow_data[:Theta_Data])
    Vr_Data    = convert(Matrix{Float64}, powerflow_data[:Vr_Data])
    Vi_Data    = convert(Matrix{Float64}, powerflow_data[:Vi_Data])
    Pinj_Data  = convert(Matrix{Float64}, powerflow_data[:Pinj_Data])
    Qinj_Data  = convert(Matrix{Float64}, powerflow_data[:Qinj_Data])
    Sft_Data   = convert(Matrix{Float64}, powerflow_data[:Sft_Data])
    Stf_Data   = convert(Matrix{Float64}, powerflow_data[:Stf_Data])
    Ift2_Data  = convert(Matrix{Float64}, powerflow_data[:Ift2_Data])
    Itf2_Data  = convert(Matrix{Float64}, powerflow_data[:Itf2_Data])

    # Split the data; training, validation, and test
    percent_train = 70
    num_samples   = size(Vm_Data,2)
    sample_split  = vcat([1],randperm(num_samples-1) .+ 1)
    train_length  = Int(round((percent_train/100)*num_samples))
    val_length    = Int(round((num_samples-train_length)/2))
    test_length   = Int(num_samples - train_length - val_length)

    # Define split indices
    train_indices = sample_split[(1):(train_length)]
    val_indices   = sample_split[(train_length+1):(train_length+val_length)]
    test_indices  = sample_split[(train_length+val_length+1):(train_length+val_length+test_length)]

    # Ensure the sizes are correct
    if (train_length+val_length+test_length) == num_samples
        println("Data split successful!")
    else
        println("Data split *NOT* successful!")
    end

    # Perform the actual split -- Training data
    Vm_TrainData    = Vm_Data[:,train_indices]
    Theta_TrainData = Theta_Data[:,train_indices]
    Vr_TrainData    = Vr_Data[:,train_indices]
    Vi_TrainData    = Vi_Data[:,train_indices]
    Pinj_TrainData  = Pinj_Data[:,train_indices]
    Qinj_TrainData  = Qinj_Data[:,train_indices]
    Sft_TrainData   = Sft_Data[:,train_indices]
    Stf_TrainData   = Stf_Data[:,train_indices]
    Ift2_TrainData  = Ift2_Data[:,train_indices]
    Itf2_TrainData  = Itf2_Data[:,train_indices]

    # Perform the actual split -- Validation data
    Vm_ValidationData    = Vm_Data[:,val_indices]
    Theta_ValidationData = Theta_Data[:,val_indices]
    Vr_ValidationData    = Vr_Data[:,val_indices]
    Vi_ValidationData    = Vi_Data[:,val_indices]
    Pinj_ValidationData  = Pinj_Data[:,val_indices]
    Qinj_ValidationData  = Qinj_Data[:,val_indices]
    Sft_ValidationData   = Sft_Data[:,val_indices]
    Stf_ValidationData   = Stf_Data[:,val_indices]
    Ift2_ValidationData  = Ift2_Data[:,val_indices]
    Itf2_ValidationData  = Itf2_Data[:,val_indices]

    # Perform the actual split -- Test data
    Vm_TestData    = Vm_Data[:,test_indices]
    Theta_TestData = Theta_Data[:,test_indices]
    Vr_TestData    = Vr_Data[:,test_indices]
    Vi_TestData    = Vi_Data[:,test_indices]
    Pinj_TestData  = Pinj_Data[:,test_indices]
    Qinj_TestData  = Qinj_Data[:,test_indices]
    Sft_TestData   = Sft_Data[:,test_indices]
    Stf_TestData   = Stf_Data[:,test_indices]
    Ift2_TestData  = Ift2_Data[:,test_indices]
    Itf2_TestData  = Itf2_Data[:,test_indices]

    # Print the splitting
    println(" - Number of training samples: "*string(length(train_indices)))
    println(" - Number of validation samples: "*string(length(val_indices)))
    println(" - Number of testing samples: "*string(length(test_indices)))

    # Write to file
    fid = h5open(powerflow_data_file, "w") do file
        # training data
        write(file, "Vm_TrainData", Vm_TrainData)
        write(file, "Theta_TrainData", Theta_TrainData)
        write(file, "Vr_TrainData", Vr_TrainData)
        write(file, "Vi_TrainData", Vi_TrainData)
        write(file, "Pinj_TrainData", Pinj_TrainData)
        write(file, "Qinj_TrainData", Qinj_TrainData)
        write(file, "Sft_TrainData", Sft_TrainData)
        write(file, "Stf_TrainData", Stf_TrainData)
        write(file, "Ift2_TrainData", Ift2_TrainData)
        write(file, "Itf2_TrainData", Itf2_TrainData)

        # validation data
        write(file, "Vm_ValidationData", Vm_ValidationData)
        write(file, "Theta_ValidationData", Theta_ValidationData)
        write(file, "Vr_ValidationData", Vr_ValidationData)
        write(file, "Vi_ValidationData", Vi_ValidationData)
        write(file, "Pinj_ValidationData", Pinj_ValidationData)
        write(file, "Qinj_ValidationData", Qinj_ValidationData)
        write(file, "Sft_ValidationData", Sft_ValidationData)
        write(file, "Stf_ValidationData", Stf_ValidationData)
        write(file, "Ift2_ValidationData", Ift2_ValidationData)
        write(file, "Itf2_ValidationData", Itf2_ValidationData)

        # test data
        write(file, "Vm_TestData", Vm_TestData)
        write(file, "Theta_TestData", Theta_TestData)
        write(file, "Vr_TestData", Vr_TestData)
        write(file, "Vi_TestData", Vi_TestData)
        write(file, "Pinj_TestData", Pinj_TestData)
        write(file, "Qinj_TestData", Qinj_TestData)
        write(file, "Sft_TestData", Sft_TestData)
        write(file, "Stf_TestData", Stf_TestData)
        write(file, "Ift2_TestData", Ift2_TestData)
        write(file, "Itf2_TestData", Itf2_TestData)
    end

    # Write network information to file
    fid = h5open(powerflow_network_file, "w") do file
        # network data -- Useful for training, bounding big-M, etc.
        write(file, "V_max", system_limits[:V_max])
        write(file, "V_min", system_limits[:V_min])
        write(file, "DTheta_max", system_limits[:DTheta_max])
        write(file, "DTheta_min", system_limits[:DTheta_min])
        write(file, "Pinj_max", system_limits[:Pinj_max])
        write(file, "Pinj_min", system_limits[:Pinj_min])
        write(file, "Qinj_max", system_limits[:Qinj_max])
        write(file, "Qinj_min", system_limits[:Qinj_min])
        write(file, "Ift2_max", system_limits[:Ift2_max])
        write(file, "Itf2_max", system_limits[:Itf2_max])
        write(file, "num_buses", system_limits[:num_buses])
        write(file, "num_lines", system_limits[:num_lines])
        write(file, "ref_bus", system_limits[:ref_bus])
        end

    return Vm_TrainData, Vm_ValidationData, Vm_TestData
end

# Call all relevant system limits
function get_system_limits(network_data::Dict,load_scaling::Float64)
    # system parameters
    num_buses = length(network_data["bus"])
    num_lines = length(network_data["branch"])
    num_gens  = length(network_data["gen"])
    num_loads = length(network_data["load"])

    # identify reference bus
    ref_bus = Float64
    for (ii,bus) in network_data["bus"]
        if bus["bus_type"] == 3
            ref_bus = parse(Int,ii)
        end
    end

    # voltage Limits
    V_max = Float64[]
    V_min = Float64[]
    for ii = 1:num_buses
        # Get votlage limits
        push!(V_max,network_data["bus"][string(ii)]["vmax"])
        push!(V_min,network_data["bus"][string(ii)]["vmin"])
    end

    # branch Limits
    Ift_max    = Float64[]
    Itf_max    = Float64[]
    Ift2_max   = Float64[]
    Itf2_max   = Float64[]
    DTheta_max = Float64[]
    DTheta_min = Float64[]
    for ii = 1:num_lines
        # Get votlage limits
        push!(Ift_max,network_data["branch"][string(ii)]["c_rating_a"])
        push!(Itf_max,network_data["branch"][string(ii)]["c_rating_a"])
        push!(Ift2_max,network_data["branch"][string(ii)]["c_rating_a"]^2)
        push!(Itf2_max,network_data["branch"][string(ii)]["c_rating_a"]^2)
        push!(DTheta_max,network_data["branch"][string(ii)]["angmax"])
        push!(DTheta_min,network_data["branch"][string(ii)]["angmin"])
    end

    # generation limits
    Pg_max = zeros(num_buses)
    Qg_max = zeros(num_buses)
    Pg_min = zeros(num_buses)
    Qg_min = zeros(num_buses)
    for ii = 1:num_gens
          bus_ind         = network_data["gen"][string(ii)]["gen_bus"]
          Pg_max[bus_ind] = Pg_max[bus_ind] + network_data["gen"][string(ii)]["pmax"]
          Qg_max[bus_ind] = Qg_max[bus_ind] + network_data["gen"][string(ii)]["qmax"]
          Pg_min[bus_ind] = Pg_min[bus_ind] + network_data["gen"][string(ii)]["pmin"]
          Qg_min[bus_ind] = Qg_min[bus_ind] + network_data["gen"][string(ii)]["qmin"]
    end

    # nominal load limits
    Pd_max = zeros(num_buses)
    Qd_max = zeros(num_buses)
    Pd_min = zeros(num_buses)
    Qd_min = zeros(num_buses)
    for ii = 1:num_loads
        bus_ind = network_data["load"][string(ii)]["load_bus"]

        # test: positive or negative P? this is confusing..
        if network_data["load"][string(ii)]["pd"] > 0
            Pd_max[bus_ind] = Pd_max[bus_ind] + (1+load_scaling)*network_data["load"][string(ii)]["pd"]
            Pd_min[bus_ind] = Pd_min[bus_ind] + (1-load_scaling)*network_data["load"][string(ii)]["pd"]
        else
            Pd_max[bus_ind] = Pd_max[bus_ind] + (1-load_scaling)*network_data["load"][string(ii)]["pd"]
            Pd_min[bus_ind] = Pd_min[bus_ind] + (1+load_scaling)*network_data["load"][string(ii)]["pd"]
        end

        # test: positive or negative Q? this is confusing..
        if network_data["load"][string(ii)]["qd"] > 0
            Qd_max[bus_ind] = Qd_max[bus_ind] + (1+load_scaling)*network_data["load"][string(ii)]["qd"]
            Qd_min[bus_ind] = Qd_min[bus_ind] + (1-load_scaling)*network_data["load"][string(ii)]["qd"]
        else
            Qd_max[bus_ind] = Qd_max[bus_ind] + (1-load_scaling)*network_data["load"][string(ii)]["qd"]
            Qd_min[bus_ind] = Qd_min[bus_ind] + (1+load_scaling)*network_data["load"][string(ii)]["qd"]
        end
    end

    # injection limits
    Pinj_max = Pg_max - Pd_min
    Pinj_min = Pg_min - Pd_max
    Qinj_max = Qg_max - Qd_min
    Qinj_min = Qg_min - Qd_max

    return Dict(:V_max => V_max, :V_min => V_min, :Ift2_max => Ift2_max, :Itf2_max => Itf2_max, :DTheta_max => DTheta_max,
                :DTheta_min => DTheta_min, :Pinj_max => Pinj_max, :Qinj_max => Qinj_max, :Pinj_min => Pinj_min, :Qinj_min => Qinj_min,
                :Ift_max => Ift_max, :Itf_max => Itf_max, :num_buses => num_buses, :num_lines => num_lines, :ref_bus => ref_bus)
end

# Parse OPF solution
function parse_OPF(OPF_soln,Yb,YLft,YLtf,YSftE1,YStfE2,E1,E2,network_data)
    num_buses = size(Yb,1)
    num_lines = size(E1,1)
    Vm        = zeros(num_buses)
    Theta     = zeros(num_buses)

    for ii = 1:num_buses
        Vm[ii]    = OPF_soln["solution"]["bus"][string(ii)]["vm"]
        Theta[ii] = OPF_soln["solution"]["bus"][string(ii)]["va"]
    end

    # current flows
    Vc   = Vm.*exp.(im*Theta)
    Ift  = (YLft+YSftE1)*Vc
    Ift2 = abs.(Ift).^2
    Itf  = (YLtf+YStfE2)*Vc
    Itf2 = abs.(Itf).^2

    # apparent power flows (f->t and t->f)
    Sft = abs.((E1*Vc).*conj(Ift))
    Stf = abs.((E2*Vc).*conj(Itf))

    # power injections
    Sc   = Vc.*conj(Yb*Vc)
    Pinj = real(Sc)
    Qinj = imag(Sc)

    # Do the current flows match the explicit branch flows?
    Pft = real((E1*Vc).*conj(Ift));
    Qft = imag((E1*Vc).*conj(Ift));
    Ptf = real((E2*Vc).*conj(Itf));
    Qtf = imag((E2*Vc).*conj(Itf));

    # loop over branches and test
    jj = 1
    for ii = 1:num_lines
        while network_data["branch"][string(jj)]["br_status"] == 0
            jj = jj + 1 # increment when a line is "off"
        end

        if abs(Pft[ii] - OPF_soln["solution"]["branch"][string(jj)]["pf"]) > 1e-8
            println("***The OPF solution parsing 1 has a problem***")
        end
        if abs(Qft[ii] - OPF_soln["solution"]["branch"][string(jj)]["qf"]) > 1e-8
            println("***The OPF solution parsing 2 has a problem***")
        end
        if abs(Ptf[ii] - OPF_soln["solution"]["branch"][string(jj)]["pt"]) > 1e-8
            println("***The OPF solution parsing 3 has a problem***")
        end
        if abs(Qtf[ii] - OPF_soln["solution"]["branch"][string(jj)]["qt"]) > 1e-8
            println("***The OPF solution parsing 4 has a problem***")
        end
        jj = jj + 1 # increment after comparing flows
    end

    # Return data
    return Vm, Theta, Pinj, Qinj, Sft, Stf, Ift2, Itf2
end

# Loop and solve IPOPT
function solve_IPOPTs(network_data::Dict,num_pf_solves::Int)

    # set to 1 until the first valid power flow solution has been found
    init_pf = 1
    
    # get useful system matrices
    Yb, ~, ~, ~, E1, E2, ~, YSftE1, YStfE2, ~, ~, YLft, YLtf = network_structures(network_data)

    # call the ref
    ref = PowerModels.build_ref(network_data)[:it][pm_it_sym][:nw][nw_id_default]

    # build a custom OPF objective function -- the key here is "_build_opf_cl"
    pm = instantiate_model(network_data, ACPPowerModel, PowerModels._build_opf_cl)

    # parse the variables
    variable_dict = pm.var[:it][:pm][:nw][0]

    # Define varibles of interest
    pg = variable_dict[:pg]
    qg = variable_dict[:qg]
    vm = variable_dict[:vm]
    va = variable_dict[:va]

    # let's randomize the starting values
    for ii in 1:length(vm)
        vm_rand = (rand(1)[1] .- 0.5)./4 .+ 1.0 # 1 +/- 0.125
        va_rand = (rand(1)[1] .- 0.5)./4 .+ 0.0 # 0 +/- 0.125
        set_start_value(vm[ii], vm_rand)
        set_start_value(va[ii], va_rand)
    end

    # identify reference bus
    ref_bus = Float64
    for (ii,bus) in network_data["bus"]
        if bus["bus_type"] == 3
            ref_bus = parse(Int,ii)
        end
    end

    # define generator indices
    gen_inds  = axes(pg)[1]
    bus_inds  = axes(vm)[1]
    num_gens  = length(gen_inds)
    num_buses = length(bus_inds)
    p_inds    = 1
    q_inds    = 1
    v_inds    = 1

    # store data as index sets
    Pgen_soln = JuMP.Containers.DenseAxisArray(zeros(num_gens,num_pf_solves),  gen_inds, 1:num_pf_solves)
    Qgen_soln = JuMP.Containers.DenseAxisArray(zeros(num_gens,num_pf_solves),  gen_inds, 1:num_pf_solves)
    Vm_soln   = JuMP.Containers.DenseAxisArray(zeros(num_buses,num_pf_solves), bus_inds, 1:num_pf_solves)

    # Hard-code some optimization parameters
    num_opt_p_vars = min(50,num_gens)  # number of variables to use in the nonlinear objective function
    num_opt_q_vars = min(50,num_gens)  # number of variables to use in the nonlinear objective function
    num_opt_v_vars = min(50,num_buses) # number of variables to use in the nonlinear objective function
    max_solns      = 35                # maximum number of pf solutions to keep in the nonlinear objective function
    T_max          = 150.0
    It_max         = 2000

    for ii = 1:num_pf_solves

        if init_pf != 1
            gen_inds_subset_p = copy(shuffle(gen_inds)[1:num_opt_p_vars])
            gen_inds_subset_q = copy(shuffle(gen_inds)[1:num_opt_q_vars]) 
            bus_inds_subset_v = copy(shuffle(bus_inds)[1:num_opt_v_vars])

            # Define a new nonlinear objective -- f38 (modified)
            @NLobjective(pm.model, Max,
               sum(log(sum(abs(pg[i]-Pgen_soln[i,j]) for i in gen_inds_subset_p)) for j in p_inds)+
               sum(log(sum(abs(qg[i]-Qgen_soln[i,j]) for i in gen_inds_subset_q)) for j in q_inds)+
               sum(log(sum(abs(vm[i]-Vm_soln[i,j])   for i in bus_inds_subset_v)) for j in v_inds))
            
            # Define a new nonlinear objective -- original formulation
            #= @NLobjective(pm.model, Max,
                sum(log(sum(abs(pg[i]-Pgen_soln[i,j]) for (i,gen) in ref[:gen])) for j in p_inds)+
                sum(log(sum(abs(qg[i]-Qgen_soln[i,j]) for (i,gen) in ref[:gen])) for j in q_inds)+
                sum(log(sum(abs(vm[i]-Vm_soln[i,j])   for (i,bus) in ref[:bus])) for j in v_inds)) =#
        end

        # Traditional OPF solve with current (*not apparent power constraints*)
            #  ----- > OPF_soln = PowerModels._solve_opf_cl(network_data, ACPPowerModel, Ipopt.Optimizer)
        OPF_soln = optimize_model!(pm, optimizer=optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time" => T_max, "max_iter" => It_max, "print_level" => 0))

        print("Solution: ")
        print(OPF_soln["termination_status"])
        println(". Iteration number: " * string(ii))

        # is the solution valid?
        if Int(OPF_soln["termination_status"]) in [1; 4; 7]

            # We have found a valid solution! parse and update the data matrices
            Pgen_soln[gen_inds,ii] = copy(value.(pg[gen_inds]).data)
            Qgen_soln[gen_inds,ii] = copy(value.(qg[gen_inds]).data)
            Vm_soln[bus_inds,ii]   = copy(value.(vm[bus_inds]).data)

            # as Pgen_soln et al. grow large, we want to keep a random subset 
            if ii > max_solns
                # grab random values
                p_inds = randperm(ii)[1:max_solns]
                q_inds = randperm(ii)[1:max_solns]
                v_inds = randperm(ii)[1:max_solns]
            else
                p_inds = 1:ii
                q_inds = 1:ii
                v_inds = 1:ii
            end

            # Now, parse the data we will actually save
            Vm, Theta, Pinj, Qinj, Sft, Stf, Ift2, Itf2 = parse_OPF(OPF_soln,Yb,YLft,YLtf,YSftE1,YStfE2,E1,E2,network_data)
            
            # rotate Solutions
            Theta = Theta .- Theta[ref_bus]

            # Comptue Cartesian coordinates
            Vc = Vm.*exp.(Theta*im)
            Vr = real(Vc)
            Vi = imag(Vc)

            # record
            if init_pf == 1
                global Vm_Data    = copy(Vm)
                global Theta_Data = copy(Theta)
                global Vr_Data    = copy(Vr)
                global Vi_Data    = copy(Vi)
                global Pinj_Data  = copy(Pinj)
                global Qinj_Data  = copy(Qinj)
                global Sft_Data   = copy(Sft)
                global Stf_Data   = copy(Stf)
                global Ift2_Data  = copy(Ift2)
                global Itf2_Data  = copy(Itf2)

                # don't enter this loop again =)
                init_pf = 0
            else
                global Vm_Data    = hcat(Vm_Data,Vm)
                global Theta_Data = hcat(Theta_Data,Theta)
                global Vr_Data    = hcat(Vr_Data,Vr)
                global Vi_Data    = hcat(Vi_Data,Vi)
                global Pinj_Data  = hcat(Pinj_Data,Pinj)
                global Qinj_Data  = hcat(Qinj_Data,Qinj)
                global Sft_Data   = hcat(Sft_Data,Sft)
                global Stf_Data   = hcat(Stf_Data,Stf)
                global Ift2_Data  = hcat(Ift2_Data,Ift2)
                global Itf2_Data  = hcat(Itf2_Data,Itf2)
            end
        end
    end

    # return a data dictionary 
    powerflow_data = Dict(:Vm_Data => Vm_Data, :Theta_Data => Theta_Data, :Vr_Data => Vr_Data, :Vi_Data => Vi_Data,
                          :Pinj_Data => Pinj_Data, :Qinj_Data => Qinj_Data, :Sft_Data => Sft_Data, :Stf_Data => Stf_Data,
                          :Ift2_Data => Ift2_Data, :Itf2_Data => Itf2_Data)

    return powerflow_data, Pgen_soln, Qgen_soln, Vm_soln
end


# Update the powermodel
function update_powermodel(network_data::Dict,load_scaling::Float64)
    num_buses = length(network_data["bus"])
    num_lines = length(network_data["branch"])
    num_gens  = length(network_data["gen"])
    num_loads = length(network_data["load"])

    # Copy the network
    network_data_corrected = deepcopy(network_data)


    # loop and eliminate apparent power flow limits
    for ii = 1:num_lines
        delete!(network_data_corrected["branch"][string(ii)], "rate_a")
        delete!(network_data_corrected["branch"][string(ii)], "rate_b")
        delete!(network_data_corrected["branch"][string(ii)], "rate_c")
    end

    # create an empty generator dictionary
    empty_gen_dict = Dict(
                "pg"         => 0.0,
                "model"      => 2,
                "shutdown"   => 0.0,
                "startup"    => 0.0,
                "qg"         => 0,
                "gen_bus"    => 0,
                "pmax"       => 0.0,
                "vg"         => 1.0,
                "mbase"      => 100.0,
                "source_id"  => 
                "index"      => 0,
                "cost"       => [],
                "qmax"       => 0.0,
                "gen_status" => 1,
                "qmin"       => 0.0,
                "pmin"       => 0.0,
                "ncost"      => 0)

    
    # loop over loads: convert each into a generator
    for ii = 1:num_loads
        bus_ind = network_data["load"][string(ii)]["load_bus"]
        if network_data["load"][string(ii)]["pd"] > 0
            Pd_max  = (1+load_scaling)*network_data["load"][string(ii)]["pd"]
            Pd_min  = (1-load_scaling)*network_data["load"][string(ii)]["pd"]
        else
            Pd_min  = (1+load_scaling)*network_data["load"][string(ii)]["pd"]
            Pd_max  = (1-load_scaling)*network_data["load"][string(ii)]["pd"]
        end

        if network_data["load"][string(ii)]["qd"] > 0
            Qd_max  = (1+load_scaling)*network_data["load"][string(ii)]["qd"]
            Qd_min  = (1-load_scaling)*network_data["load"][string(ii)]["qd"]
        else
            Qd_min  = (1+load_scaling)*network_data["load"][string(ii)]["qd"]
            Qd_max  = (1-load_scaling)*network_data["load"][string(ii)]["qd"]
        end

        # assign to a new generator
        new_load_generator              = deepcopy(empty_gen_dict)
        new_load_generator["pmax"]      = -Pd_min      # these are flipped!!
        new_load_generator["pmin"]      = -Pd_max      # these are flipped!!

        new_load_generator["qmax"]      = -Qd_min      # these are flipped!!
        new_load_generator["qmin"]      = -Qd_max      # these are flipped!!

        new_load_generator["gen_bus"]   = bus_ind
        new_load_generator["index"]     = ii + num_gens
        new_load_generator["source_id"] = Any["gen", ii + num_gens]

        # update
        network_data_corrected["gen"][string(ii + num_gens)] = new_load_generator

        # now, delete the load
        delete!(network_data_corrected["load"],string(ii))
    end

    return network_data_corrected
end