
function run_strongSDP_verification_individual_mats(ac_power_flow_model::ac_power_flow_model_instance,ac_nn_model::ac_nn_model_instance,verication_routine::String,verification_index::Int,include_constraints::Vector,eta_PSD::Bool=false,strong_SDP::Bool=false)
    # This (depricated function keeps smaller matrices PSD, rather than all of them)
    
    # Build model
    model = Model(Mosek.Optimizer)
    empty!(model)

    # Set attibutes?
          # set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-5)
          # set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_MU_RED", 1e-5)

    # Call NN parameters
    W0       = ac_nn_model.W0
    b0       = ac_nn_model.b0
    W1       = ac_nn_model.W1
    b1       = ac_nn_model.b1
    M_max    = ac_nn_model.M_max
    M_min    = ac_nn_model.M_min
    In_Mean  = ac_nn_model.In_Mean
    In_STD   = ac_nn_model.In_STD
    Out_Mean = ac_nn_model.Out_Mean
    Out_STD  = ac_nn_model.Out_STD
    J0       = ac_nn_model.J0
    r0       = ac_nn_model.r0

    # Call power flow constraints
    V_max    = ac_power_flow_model.V_max
    V_min    = ac_power_flow_model.V_min
    V2_max   = V_max.^2
    V2_min   = V_min.^2
    Ift2_max = ac_power_flow_model.Ift2_max
    Itf2_max = ac_power_flow_model.Itf2_max
    Ift2_min = 0*Ift2_max
    Itf2_min = 0*Itf2_max
    PB_max   = ac_power_flow_model.PB_max
    PB_min   = ac_power_flow_model.PB_min
    QB_max   = ac_power_flow_model.QB_max
    QB_min   = ac_power_flow_model.QB_min
    V_maxmax = maximum(V_max)

    # Define system parameters
    num_buses = ac_power_flow_model.num_buses
    num_lines = ac_power_flow_model.num_lines
    ref_bus   = ac_power_flow_model.ref_bus
    num_ReLU  = length(b0)

    bus_map = collect([1:num_buses;1:num_buses])
    deleteat!(bus_map,num_buses+ref_bus)

    # Define the SDP Matrices
    @variable(model,Binary_SDP_full[1:(num_ReLU+1), 1:(num_ReLU+1)], PSD)
    @variable(model,Voltage_SDP_full[1:(2*num_buses), 1:(2*num_buses)], PSD)

    # Constrain eta to be PSD?
    if eta_PSD == true
          @variable(model,X_SDP_full[1:(num_ReLU+1), 1:(num_ReLU+1)], PSD)
          @constraint(model, X_SDP_full[1,1]  == 1)
          eta = X_SDP_full[2:end,2:end]
    else
          @variable(model,eta[1:(num_ReLU+1),1:(num_ReLU+1)],Symmetric)
    end

    # Constrain the leading entry to be one
    @constraint(model, Binary_SDP_full[1,1]  == 1)
    @constraint(model, Voltage_SDP_full[1,1] == 1)

    # Add binary constraints
    B    = Binary_SDP_full[2:end,2:end]
    beta = Binary_SDP_full[1,2:end]
    @constraint(model, diag(B) .== beta)
    @constraint(model, 0 .<= Binary_SDP_full[:, :] .<= 1)

    # Constrain diagonals to be greater than or equal to the corresponding off-diagonals
    for ii = 1:num_ReLU
          for jj = 1:num_ReLU
                if ii <= jj # B is symmetric
                      if ii != jj
                            @constraint(model, B[ii,jj] <= B[ii,ii]) 
                            @constraint(model, B[ii,jj] <= B[jj,jj])
                      end
                end
          end
    end

    # Constrain off-diagonals to be greater than the "sum minus 1" of the corresponding diagonals
    for ii = 1:num_ReLU
          for jj = 1:num_ReLU
                if ii <= jj # B is symmetric
                      if ii != jj
                            @constraint(model, (B[ii,ii] + B[jj,jj] - 1) <= B[ii,jj])
                      end
                end
          end
    end

    # Add voltage constraints
    v = Voltage_SDP_full[1,2:end]
    V = Voltage_SDP_full[2:end,2:end]
    @constraint(model, -(V_maxmax^2) .<= V[:, :] .<= (V_maxmax^2))

    # Define the NN constraints: build the A matrix
    W0_bar = W0*Diagonal(vec(1 ./In_STD))  # W0.*(1 ./In_STD)'
    b0_bar = b0 - W0_bar*In_Mean
    W1_bar = Out_STD.*W1
    b1_bar = Out_STD.*b1 + Out_Mean + r0
    M_maxD = Diagonal(M_max)
    M_minD = Diagonal(M_min)

    # Define A sub-matrices
    nv = 2*num_buses - 1
    nx = num_ReLU
    nb = num_ReLU
    no = length(r0)
    Ix = sparse(I,nx,nx)
    Zb = spzeros(nb,nb)
    Zv = spzeros(nb,nv)
    A0nn = [W0_bar -Ix   Zb;
            Zv     -Ix   Zb;
           -W0_bar  Ix  -M_minD;
            Zv      Ix  -M_maxD]
    b0nn = [-b0_bar;
             zeros(nx);
             b0_bar - M_min;
             zeros(nx)]
    AJ  = [J0; -J0]
    Apf = [ sparse(I,no,no);
           -sparse(I,no,no)]
    bpf = [V2_max;
          PB_max;
          QB_max;
          Ift2_max;
          Itf2_max;
          -(V2_min);
          -(PB_min);
          -(QB_min);
          -(Ift2_min);
          -(Itf2_min)]

    # Define the A matrix and the b vector
    A = [A0nn;
         AJ Apf*W1_bar spzeros(2*no,nb)]
    b = [b0nn;
         bpf-Apf*b1_bar]
    
    # Define the ReLU output (x)
    @variable(model,0 <= x[1:num_ReLU])
    @constraint(model,x[1:num_ReLU] .<= M_max)
    if eta_PSD == true
          @constraint(model, X_SDP_full[1,2:end] .== x)
    end

    u = [v; x; beta]

    # Add nn and power flow constraints
    @constraint(model, A*u .<= b)

    # Define an ouput vector and parse
    @expression(model, y, W1_bar*x + J0*v + b1_bar) # y = W1_bar*x + b1_bar + J0*v
    Vm2_nn  = y[                             (1:num_buses)]
    P_nn    = y[num_buses                 .+ (1:num_buses)]
    Q_nn    = y[2*num_buses               .+ (1:num_buses)]
    Ift2_nn = y[3*num_buses               .+ (1:num_lines)]
    Itf2_nn = y[3*num_buses + 1*num_lines .+ (1:num_lines)]

    # Strengthen the SDP formulation?
    if strong_SDP == true

          # Define gamma = vbt
          @variable(model,gamma[1:nv,1:nb])
          for ii = 1:nv
                for jj = 1:nb
                      bus_index = bus_map[ii]
                      M_max_vb =  V_max[bus_index]
                      M_min_vb = -V_max[bus_index]
                      @constraint(model,gamma[ii,jj] <= v[ii] - M_min_vb*(1-beta[jj]))
                      @constraint(model,v[ii] - M_max_vb*(1-beta[jj]) <= gamma[ii,jj])
                      @constraint(model,gamma[ii,jj] <= M_max_vb*beta[jj])
                      @constraint(model,M_min_vb*beta[jj] <= gamma[ii,jj])
                end
          end

          # Define delta = xbt
          @variable(model,delta[1:nx,1:nb])
          for ii = 1:nx
                for jj = 1:nb
                      M_max_xb = M_max[ii]
                      M_min_xb = 0
                      @constraint(model,delta[ii,jj] <= x[ii])
                      @constraint(model,x[ii] - M_max_xb*(1-beta[jj]) <= delta[ii,jj])
                      @constraint(model,delta[ii,jj] <= M_max_xb*beta[jj])
                      @constraint(model,0 <= delta[ii,jj])
                end
          end

          # Define epsilon = vxt
          @variable(model,epsilon[1:nv,1:nx])
          for ii = 1:nv
                for jj = 1:nx
                      bus_index = bus_map[ii]
                      zij = AffExpr(0.0)  # zij = W0_bar[jj,:]'*V[:,ii] + v[ii]*b0_bar[jj]
                      add_to_expression!(zij,v[ii]*b0_bar[jj])
                      add_to_expression!(zij,W0_bar[jj,:]'*V[:,ii])

                      M_max_vx = maximum((V_max[bus_index]*M_max[jj],-V_max[bus_index]*M_min[jj]))
                      M_min_vx = minimum((V_max[bus_index]*M_min[jj],-V_max[bus_index]*M_max[jj]))
                      @constraint(model,epsilon[ii,jj] <= zij - M_min_vx*(1-beta[jj]))
                      @constraint(model,zij - M_max_vx*(1-beta[jj]) <= epsilon[ii,jj])
                      @constraint(model,epsilon[ii,jj] <= M_max_vx*beta[jj])
                      @constraint(model,M_min_vx*beta[jj] <= epsilon[ii,jj])
                end
          end

          # Define (call) eta = xxt
          for ii = 1:nx
                for jj = 1:nx
                      xi_ij = AffExpr(0.0) # xi_ij = W0_bar[ii,:]'*V*(W0_bar[jj,:]) + W0_bar[jj,:]'*v*b0_bar[ii] + W0_bar[ii,:]'*v*b0_bar[jj] + b0_bar[ii]*b0_bar[jj]
                      add_to_expression!(xi_ij,b0_bar[ii]*b0_bar[jj])
                      add_to_expression!(xi_ij,W0_bar[ii,:]'*v*b0_bar[jj])
                      add_to_expression!(xi_ij,W0_bar[jj,:]'*v*b0_bar[ii])
                      add_to_expression!(xi_ij,W0_bar[ii,:]'*V*(W0_bar[jj,:]))

                      M_max_xx = maximum((M_max[ii]*M_max[jj],M_min[ii]*M_min[jj]))
                      M_min_xx = minimum((M_max[ii]*M_min[jj],M_min[ii]*M_max[jj]))
                      @constraint(model,eta[ii,jj] <= xi_ij - M_min_xx*(1-B[ii,jj]))
                      @constraint(model,xi_ij - M_max_xx*(1-B[ii,jj]) <= eta[ii,jj])
                      @constraint(model,eta[ii,jj] <= M_max_xx*B[ii,jj])
                      @constraint(model,0 <= eta[ii,jj])
                end
          end
          # Finally, we need to build up the "Gamma = (u)(u^T)=uut" matrix
          # Gamma = [V    vxt   vbt;
          #          vxt' xxt   xbt;
          #          vbt' xbt'   B]
          Gamma = [V          epsilon   gamma;
                   epsilon'   eta       delta;
                   gamma'     delta'    B]

          # Loop to define the Omega matrix, but skip constraints we will not use
                # Alternative:  ->   Omega = A*Gamma*(A') - A*u*(b') - b*(A*u)' + b*b'
          @expression(model, GAp, Gamma*(A'))
          @expression(model, Au, A*u)

          for ii = 1:size(A,1)
                for jj = 1:size(A,1)
                      if ii <= jj
                            Omgea_ij = AffExpr(0.0)
                            add_to_expression!(Omgea_ij,b[ii]*b[jj])
                            add_to_expression!(Omgea_ij,-b[ii]*Au[jj])
                            add_to_expression!(Omgea_ij,-Au[ii]*b[jj])
                            add_to_expression!(Omgea_ij,A[ii,:]'*GAp[:,jj])
                            if (ii,jj) in include_constraints
                                  @constraint(model, 0 .<= Omgea_ij)
                            end
                      end
                end
          end
    end

    # Next, add the quadratic equality constraints:
    for ii = 1:nx
          @constraint(model,eta[ii,ii]  ==  W0_bar[ii,:]'*epsilon[:,ii] + x[ii]*b0_bar[ii])
    end

    ###########################################################
    # Build Verification Objective Functions ##################
    ###########################################################
    Verification_Results = Dict()
    #
    # In this case, find the maximum error for a single line/bus
    if verification_index > 0
          # Test the type of verification
          if verication_routine == "Squared_Voltage_Magnitude"
                bus      = verification_index
                Mat      = ac_power_flow_model.bus_Vmag2_QMat[bus][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                obj_up   = Vm2_nn[bus] - tr(Mat*V) # transformed from v'*Mat_ft*v
                obj_down = -obj_up

          ###########################################
          elseif verication_routine == "P_Injection"
                bus      = verification_index
                Mat      = ac_power_flow_model.bus_Pinj_QMat[bus][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                obj_up   = P_nn[bus] - tr(Mat*V)
                obj_down = -obj_up

          ###########################################
          elseif verication_routine == "Q_Injection"
                bus      = verification_index
                Mat      = ac_power_flow_model.bus_Qinj_QMat[bus][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                obj_up   = Q_nn[bus] - tr(Mat*V)
                obj_down = -obj_up

          ###########################################
          elseif verication_routine == "Squared_Current_ft"
                line     = verification_index
                Mat_ft   = ac_power_flow_model.line_Ift2_QMat[line][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                obj_up   = Ift2_nn[line] - tr(Mat_ft*V)
                obj_down = -obj_up

          elseif verication_routine == "Squared_Current_tf"
                line     = verification_index
                Mat_tf   = ac_power_flow_model.line_Itf2_QMat[line][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                obj_up   = Itf2_nn[line] - tr(Mat_tf*V)
                obj_down = -obj_up

          else
                println("Verification routine not recognized!")
          end

          # Solve down
          @objective(model, Max, obj_down)
          optimize!(model)
          println("--------------")
          println(termination_status(model))
          println("--------------")
          println(value(obj_down));
          println("--------------")

          if strong_SDP == true
                # Define outputs
                Results = Dict(
                      :obj     => objective_value(model),
                      :B       => value.(B),
                      :V       => value.(V),
                      :x       => value.(x),
                      :v       => value.(v),
                      :beta    => value.(beta),
                      :eta     => value.(eta),
                      :epsilon => value.(epsilon),
                      :gamma   => value.(gamma),
                      :delta   => value.(delta),
                      :X_SDP   => value.(X_SDP_full),
                      :A       => value.(A),
                      :b       => value.(b),
                      :u       => value.(u))
          else
                # Define outputs
                Results = Dict(
                      :obj     => objective_value(model),
                      :B       => value.(B),
                      :V       => value.(V),
                      :x       => value.(x),
                      :v       => value.(v),
                      :beta    => value.(beta))
          end

          # Verification_Results = record_results(Verification_Results,verication_routine,model,obj_up,vr,vi,verification_index,verification_index)

    else
          println("Verification index not yet recognized!")
    end

    return Results
end

# Loop and solve IPOPT
function solve_IPOPTs_trivial_inds(network_data::Dict,num_pf_solves::Int)
    # Pgen_soln = hcat(Pgen_soln, value.(pg)[1:length(pg)].data) => this is broken :)

    # Hard-code some optimization parameters
    T_max  = 150.0
    It_max = 500

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

    # master loop
    Pgen_soln_subset = []
    Qgen_soln_subset = []
    Vm_soln_subset   = []

    for ii = 1:num_pf_solves

        if init_pf != 1
            # Define a new nonlinear objective
            @NLobjective(pm.model, Max,
                sum(log(sum(abs(pg[i]-Pgen_soln_subset[i,j]) for (i,gen) in ref[:gen])) for j in 1:size(Pgen_soln_subset,2))+
                sum(log(sum(abs(qg[i]-Qgen_soln_subset[i,j]) for (i,gen) in ref[:gen])) for j in 1:size(Qgen_soln_subset,2))+
                sum(log(sum(abs(vm[i]-Vm_soln_subset[i,j])   for (i,bus) in ref[:bus])) for j in 1:size(Vm_soln_subset,2)))
        end

        # Traditional OPF solve with current (*not apparent power constraints*)
            #  ----- > OPF_soln = PowerModels._solve_opf_cl(network_data, ACPPowerModel, Ipopt.Optimizer)
        OPF_soln = optimize_model!(pm, optimizer=optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time" => T_max, "max_iter" => It_max, "print_level" => 0))

        print("Solution: ")
        print(OPF_soln["termination_status"])
        println(". Iteration numer: " * string(ii))

        # is the solution valid?
        if Int(OPF_soln["termination_status"]) in [1; 4; 7]

            # We have found a valid solution! parse and update the data matrices:
            if init_pf == 1
                # Parse the solution and initialize data matrices -- should be in the correct order
                global Pgen_soln = copy(value.(pg)[1:length(pg)].data)
                global Qgen_soln = copy(value.(qg)[1:length(qg)].data)
                global Vm_soln   = copy(value.(vm)[1:length(vm)].data)
            else
                Pgen_soln = hcat(Pgen_soln, value.(pg)[1:length(pg)].data)
                Qgen_soln = hcat(Qgen_soln, value.(qg)[1:length(qg)].data)
                Vm_soln   = hcat(Vm_soln,   value.(vm)[1:length(vm)].data)
            end

            # as Pgen_soln et al. grow large, we want to keep a random subset 
            num_solns = size(Pgen_soln,2)
            max_solns = 50
            if num_solns > max_solns
                # grab 50 random values
                p_inds           = randperm(num_solns)[1:max_solns]
                q_inds           = randperm(num_solns)[1:max_solns]
                v_inds           = randperm(num_solns)[1:max_solns]
                Pgen_soln_subset = copy(Pgen_soln[:,p_inds])
                Qgen_soln_subset = copy(Qgen_soln[:,q_inds])
                Vm_soln_subset   = copy(Vm_soln[:,v_inds])
            else
                Pgen_soln_subset = copy(Pgen_soln)
                Qgen_soln_subset = copy(Qgen_soln)
                Vm_soln_subset   = copy(Vm_soln)
            end

            # Now, parse the data we will actually save
            Vm, Theta, Pinj, Qinj, Sft, Stf, Ift2, Itf2 = parse_OPF(OPF_soln,Yb,YLft,YLtf,YSftE1,YStfE2,E1,E2)
            
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

# append!(include_constraints,new_indices)
                  #= readline() doesn't work!!! don't mess with it :)
                  name = readline()
                  if name == "y"
                        println("this works")
                        # run again!!! remove the lower triangle
                        nOmega             = size(Omega,1)
                        Omega_UD           = Omega - [ x>y ? Omega[x,y] : 0 for x in 1:nOmega, y in 1:nOmega]
                        additional_indices = Tuple.(argsmallest(Omega_UD,new_constraints))
                        append!(include_constraints,additional_indices)
                        # Re-run!
                        run_strongSDP_verification(ac_power_flow_model,ac_nn_model,verication_routine,verification_index,include_constraints,strong_SDP,new_constraints)
                  end
                  elseif name == 0
                        # terminate
                  else
                        # terminate
                  end
                  =#
if ac_power_flow_model.num_buses == 200
      Ift2_min = -1e-1*ones(length(Ift2_max),1) # previously: 0*Ift2_max --- -1e-1 for numerics
      Itf2_min = -1e-1*ones(length(Ift2_max),1) # previously: 0*Itf2_max --- -1e-1 for numerics
else
      Ift2_min = 0*Ift2_max
      Itf2_min = 0*Itf2_max
end

# %% Run a test: does a MILP solution agree with the constraints?

# build "ac_power_flow_model"
ac_power_flow_model = build_acpf_model(network_data, powerflow_network_file)

# build "nn_model"
ac_nn_model = build_nn_model(nn_parameter_file)

# Build model
model = Model(Gurobi.Optimizer)
empty!(model)

# Call nn Parameters
W0       = ac_nn_model.W0
b0       = ac_nn_model.b0
W1       = ac_nn_model.W1
b1       = ac_nn_model.b1
M_max    = ac_nn_model.M_max
M_min    = ac_nn_model.M_min
In_Mean  = ac_nn_model.In_Mean
In_STD   = ac_nn_model.In_STD
Out_Mean = ac_nn_model.Out_Mean
Out_STD  = ac_nn_model.Out_STD
J0       = ac_nn_model.J0
r0       = ac_nn_model.r0

# Call power flow constraints
V_max    = ac_power_flow_model.V_max
V_min    = ac_power_flow_model.V_min
V2_max   = V_max.^2
V2_min   = V_min.^2
Ift2_max = ac_power_flow_model.Ift2_max
Itf2_max = ac_power_flow_model.Itf2_max
if ac_power_flow_model.num_buses == 200
    Ift2_min = -1e-1*ones(length(Ift2_max),1) # previously: 0*Ift2_max --- -1e-1 for numerics
    Itf2_min = -1e-1*ones(length(Ift2_max),1) # previously: 0*Itf2_max --- -1e-1 for numerics
else
    Ift2_min = 0*Ift2_max
    Itf2_min = 0*Itf2_max
end
Pinj_max = ac_power_flow_model.Pinj_max
Pinj_min = ac_power_flow_model.Pinj_min
Qinj_max = ac_power_flow_model.Qinj_max
Qinj_min = ac_power_flow_model.Qinj_min

# Define system parameters
num_buses = ac_power_flow_model.num_buses
num_lines = ac_power_flow_model.num_lines
ref_bus   = ac_power_flow_model.ref_bus
num_ReLU  = length(b0)

# Build input variables (bounded)
@variable(model,vr[1:num_buses])
@variable(model,vi[1:num_buses-1])
@constraint(model, -V_max               .<= vr[1:num_buses]   .<= V_max)
@constraint(model, -V_max[Not(ref_bus)] .<= vi[1:num_buses-1] .<= V_max[Not(ref_bus)])

# This is the true MILP
@variable(model, beta[1:num_ReLU], binary = true)      # Binary variable
@variable(model, 0 <= x_ReLU_out[1:num_ReLU])          # ReLU output

# set gap
set_optimizer_attribute(model, "OutputFlag", 1)

# Apply input transformation & first layer
nn_input        = [vr; vi]
nn_input_normal = (nn_input - In_Mean)./In_STD
x_ReLU_in       = W0*nn_input_normal + b0

# Constrain System of equations
for jk = 1:num_ReLU
      @constraint(model, x_ReLU_out[jk] <= x_ReLU_in[jk]-M_min[jk]*(1-beta[jk]))
      @constraint(model, x_ReLU_out[jk] <= M_max[jk]*beta[jk])
end

# Last constraint
@constraint(model, x_ReLU_in .<= x_ReLU_out)

# nn output
nn_output_normal = W1*x_ReLU_out + b1
nn_output_raw    = Out_STD.*nn_output_normal + Out_Mean

# Add linear predction
nn_output = nn_output_raw + J0*nn_input + r0

# Power Flow Prediction
Vm2_nn  = nn_output[1:num_buses]
P_nn    = nn_output[(num_buses+1):2*num_buses]
Q_nn    = nn_output[(2*num_buses+1):3*num_buses]
Ift2_nn = nn_output[(3*num_buses+1):(3*num_buses+num_lines)]
Itf2_nn = nn_output[(3*num_buses+num_lines+1):(3*num_buses+2*num_lines)]

# Add constraints to nn predictions
@constraint(model, V2_min   .<= Vm2_nn[1:num_buses]  .<= V2_max)
@constraint(model, Pinj_min .<= P_nn[1:num_buses]    .<= Pinj_max)
@constraint(model, Qinj_min .<= Q_nn[1:num_buses]    .<= Qinj_max)
@constraint(model, Ift2_min .<= Ift2_nn[1:num_lines] .<= Ift2_max)
@constraint(model, Itf2_min .<= Itf2_nn[1:num_lines] .<= Itf2_max)

@objective(model, Max, 0)
#set_optimizer_attribute(model, "OutputFlag", 0)
optimize!(model)

# %% parse
x_val     = value.(nn_input)
x_hat_val = value.(x_ReLU_out)
beta_val  = value.(beta)

u = [x_val; x_hat_val; beta_val]
Gamma = u*u'
Gamma_hat = [1 u';
             u Gamma]

# %% Compare to constraints!
# Call NN parameters
W0       = sparse(ac_nn_model.W0)
b0       = ac_nn_model.b0
W1       = sparse(ac_nn_model.W1)
b1       = ac_nn_model.b1
M_max    = ac_nn_model.M_max
M_min    = ac_nn_model.M_min
In_Mean  = ac_nn_model.In_Mean
In_STD   = ac_nn_model.In_STD
Out_Mean = ac_nn_model.Out_Mean
Out_STD  = ac_nn_model.Out_STD
J0       = ac_nn_model.J0
r0       = ac_nn_model.r0
#r0[abs.(r0) .< 1e-6] .= 0

# Call power flow constraints
V_max    = ac_power_flow_model.V_max
V_min    = ac_power_flow_model.V_min
V2_max   = V_max.^2
V2_min   = V_min.^2
Ift2_max = ac_power_flow_model.Ift2_max
Itf2_max = ac_power_flow_model.Itf2_max
if ac_power_flow_model.num_buses == 200
      Ift2_min = -1e-1*ones(length(Ift2_max),1) # previously: 0*Ift2_max --- -1e-1 for numerics
      Itf2_min = -1e-1*ones(length(Ift2_max),1) # previously: 0*Itf2_max --- -1e-1 for numerics
  else
      Ift2_min = 0*Ift2_max
      Itf2_min = 0*Itf2_max
  end
PB_max   = ac_power_flow_model.Pinj_max
PB_min   = ac_power_flow_model.Pinj_min
QB_max   = ac_power_flow_model.Qinj_max
QB_min   = ac_power_flow_model.Qinj_min

# Define system parameters
num_buses = ac_power_flow_model.num_buses
num_lines = ac_power_flow_model.num_lines
ref_bus   = ac_power_flow_model.ref_bus
num_ReLU  = length(b0)

# Bus list
bus_map = collect([1:num_buses;1:num_buses])
deleteat!(bus_map,num_buses+ref_bus)

# Build voltage constraint vectors
V_max_full = [V_max; V_max]
V_min_full = [V_min; V_min]
V_max_full = V_max_full[bus_map]
V_min_full = V_min_full[bus_map]

# Define the NN constraints: build the A matrix
W0_bar = W0*Diagonal(vec(1 ./In_STD))  # W0.*(1 ./In_STD)'
b0_bar = b0 - W0_bar*In_Mean
W1_bar = Out_STD.*W1
b1_bar = Out_STD.*b1 + Out_Mean + r0
M_maxD = Diagonal(M_max)
M_minD = Diagonal(M_min)

# Define A sub-matrices
nv = 2*num_buses - 1
nx = num_ReLU
nb = num_ReLU
no = length(r0)
Ix = sparse(I,nx,nx)
Iv = sparse(I,nv,nv)
Ib = sparse(I,nb,nb)
Zb = spzeros(nb,nb)
Zv = spzeros(nb,nv)
Zxb= spzeros(nv,nx+nb)
Zvx= spzeros(nb,nv+nx)
Ab = [-Iv Zxb;
       Iv Zxb;
       Zvx -Ib;
       Zvx  Ib]
bb = [V_max_full;
      V_max_full;
      zeros(nb);
      ones(nb)] 
A0nn = [W0_bar -Ix   Zb;
        Zv     -Ix   Zb;
       -W0_bar  Ix  -M_minD;
        Zv      Ix  -M_maxD]
b0nn = [-b0_bar;
         zeros(nx);
         b0_bar - M_min;
         zeros(nx)]
AJ  = [J0; -J0]
Apf = [ sparse(I,no,no);
       -sparse(I,no,no)]
bpf = [V2_max;
      PB_max;
      QB_max;
      Ift2_max;
      Itf2_max;
      -(V2_min);
      -(PB_min);
      -(QB_min);
      -(Ift2_min);
      -(Itf2_min)]

# Define the A matrix and the b vector
A = [Ab;
     A0nn;
     AJ Apf*W1_bar spzeros(2*no,nb)]
b = [bb;
     b0nn;
     bpf-Apf*b1_bar]

Omega = A*Gamma*(A') - A*u*(b') - b*(A*u)' + b*b'

# %% Define sub-variables
v    = Gamma_hat[1,(2)                          : (2*num_buses)             ]
x    = Gamma_hat[1,(2*num_buses + 1)            : (2*num_buses + num_ReLU)  ]
beta = Gamma_hat[1,(2*num_buses + num_ReLU + 1) : (2*num_buses + 2*num_ReLU)] 
u    = [v; x; beta]

# Define submatrices
V       = Gamma_hat[(2)                          : (2*num_buses)             , (2)                          : (2*num_buses)             ]
eta     = Gamma_hat[(2*num_buses + 1)            : (2*num_buses + num_ReLU)  , (2*num_buses + 1)            : (2*num_buses + num_ReLU)  ]
B       = Gamma_hat[(2*num_buses + num_ReLU + 1) : (2*num_buses + 2*num_ReLU), (2*num_buses + num_ReLU + 1) : (2*num_buses + 2*num_ReLU)] 
epsilon = Gamma_hat[(2)                          : (2*num_buses)             , (2*num_buses + 1)            : (2*num_buses + num_ReLU)  ]
gamma   = Gamma_hat[(2)                          : (2*num_buses)             , (2*num_buses + num_ReLU + 1) : (2*num_buses + 2*num_ReLU)] 
delta   = Gamma_hat[(2*num_buses + 1)            : (2*num_buses + num_ReLU)  , (2*num_buses + num_ReLU + 1) : (2*num_buses + 2*num_ReLU)]

# test 1
A*u - b

# %% test 2
diag(B) - beta

# test 3
tt = zeros(200,1)
ij = 1
for ii = 1:nx
    tt[ij] = abs(delta[ii,ii] - x[ii])
    ij = ij + 1
    tt[ij] = abs(eta[ii,ii] -  (W0_bar[ii,:]'*epsilon[:,ii] + x[ii]*b0_bar[ii]))
    ij = ij + 1
end

# %%
Gamma = u*u'

Gamma_num = Results[:Gamma]
u_num = Results[:u] 

Omega = A*Gamma_num*(A') - A*u_num*(b') - b*(A*u_num)' + b*b'

# %%
GAp = Gamma_num*(A')
Au  = A*u_num
kk  = 1
all_omega = zeros(200,1)

#ii_vec = zeros(length(all_constraints),1)
#jj_vec = zeros(length(all_constraints),1)

# %%

for ii = 1:size(A,1)
    for jj = 1:size(A,1)
          if ii <= jj
            if (ii,jj) in all_constraints
                      Omgea_ij = 0
                      Omgea_ij = Omgea_ij + b[ii]*b[jj]
                      Omgea_ij = Omgea_ij -b[ii]*Au[jj]
                      Omgea_ij = Omgea_ij -Au[ii]*b[jj]
                      Omgea_ij = Omgea_ij + A[ii,:]'*GAp[:,jj]
                      all_omega[kk] = Omgea_ij

                      kk = kk + 1
            end
          end
    end
end









# %% ============================
model = Model(Mosek.Optimizer)
empty!(model)

# Set attibutes?
set_optimizer_attribute(model, "QUIET", false)
set_optimizer_attribute(model, "MSK_IPAR_LOG_CUT_SECOND_OPT", 0)
# set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-5)
# set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_MU_RED", 1e-5)

# parse
initial_constraints  = verification_params[:initial_constraints]
verification_routine = verification_params[:verification_routine]
verification_index   = verification_params[:verification_index]
error_type           = verification_params[:error_type]
sparse_SDP           = verification_params[:sparse_SDP]

# Call NN parameters
W0       = sparse(ac_nn_model.W0)
b0       = ac_nn_model.b0
W1       = sparse(ac_nn_model.W1)
b1       = ac_nn_model.b1
M_max    = ac_nn_model.M_max# .+ 1e-5
M_min    = ac_nn_model.M_min# .- 1e-5
In_Mean  = ac_nn_model.In_Mean
In_STD   = ac_nn_model.In_STD
Out_Mean = ac_nn_model.Out_Mean
Out_STD  = ac_nn_model.Out_STD
J0       = ac_nn_model.J0
r0       = ac_nn_model.r0

# call the threshold for setting model parameters to 0
setzero_threshold = ac_nn_model.setzero_threshold

# Call power flow constraints
V_max    = ac_power_flow_model.V_max
V_min    = ac_power_flow_model.V_min
V2_max   = V_max.^2
V2_min   = V_min.^2
Ift2_max = ac_power_flow_model.Ift2_max
Itf2_max = ac_power_flow_model.Itf2_max
if ac_power_flow_model.num_buses == 200
    Ift2_min = -1e-1*ones(length(Ift2_max),1) # previously: 0*Ift2_max --- -1e-1 for numerics
    Itf2_min = -1e-1*ones(length(Ift2_max),1) # previously: 0*Itf2_max --- -1e-1 for numerics
else
    Ift2_min = 0*Ift2_max
    Itf2_min = 0*Itf2_max
end
PB_max   = ac_power_flow_model.Pinj_max# .+ 1e-5
PB_min   = ac_power_flow_model.Pinj_min# .- 1e-5
QB_max   = ac_power_flow_model.Qinj_max# .+ 1e-5
QB_min   = ac_power_flow_model.Qinj_min# .- 1e-5

# Define system parameters
num_buses = ac_power_flow_model.num_buses
num_lines = ac_power_flow_model.num_lines
ref_bus   = ac_power_flow_model.ref_bus
num_ReLU  = length(b0)

# Bus list
bus_map = collect([1:num_buses;1:num_buses])
deleteat!(bus_map,num_buses+ref_bus)

# Build voltage constraint vectors
V_max_full = [V_max; V_max]
V_min_full = [V_min; V_min]
V_max_full = V_max_full[bus_map]
V_min_full = V_min_full[bus_map]

# Define the NN constraints: build the A matrix
W0_bar = W0*Diagonal(vec(1 ./In_STD))  # W0.*(1 ./In_STD)'
b0_bar = b0 - W0_bar*In_Mean
W1_bar = Out_STD.*W1
b1_bar = Out_STD.*b1 + Out_Mean + r0
M_maxD = Diagonal(M_max)
M_minD = Diagonal(M_min)

# Define A sub-matrices
nv = 2*num_buses - 1
nx = num_ReLU
nb = num_ReLU
no = length(r0)
Ix = sparse(I,nx,nx)
Iv = sparse(I,nv,nv)
Ib = sparse(I,nb,nb)
Zb = spzeros(nb,nb)
Zv = spzeros(nb,nv)
Zxb= spzeros(nv,nx+nb)
Zvx= spzeros(nb,nv+nx)
Ab = [-Iv Zxb;
        Iv Zxb;
        Zvx -Ib;
        Zvx  Ib]
bb = [V_max_full;
    V_max_full;
    zeros(nb);
    ones(nb)] 
A0nn = [W0_bar -Ix   Zb;
        Zv     -Ix   Zb;
        -W0_bar  Ix  -M_minD;
        Zv      Ix  -M_maxD]
b0nn = [-b0_bar;
        zeros(nx);
        b0_bar - M_min;
        zeros(nx)]
AJ  = [J0; -J0]
Apf = [ sparse(I,no,no);
        -sparse(I,no,no)]
bpf = [V2_max;
       PB_max;
       QB_max;
       Ift2_max;
       Itf2_max;
    -(V2_min);
    -(PB_min);
    -(QB_min);
    -(Ift2_min);
    -(Itf2_min)]

# Define the A matrix and the b vector
A = [Ab;
    A0nn;
    AJ Apf*W1_bar spzeros(2*no,nb)]
b = [bb;
    b0nn;
    bpf-Apf*b1_bar]

@variable(model,v[1:(2*num_buses)-1])
@variable(model,x[1:num_ReLU])
@variable(model,beta[1:num_ReLU], binary = true)
u    = [v; x; beta]

#@variable(model, u[1:(num_ReLU + num_ReLU + 2*num_buses - 1)])

@constraint(model, A*u .<= b)

@objective(model, Max, 0)

optimize!(model)

# %%
ytst = W1_bar*value.(x) + J0*value.(v) + b1_bar


# %%
Gamma_hat = [1         value.(u)';
             value.(u) value.(u)*value.(u)'];


# Define submatrices
V       = Gamma_hat[(2)                          : (2*num_buses)             , (2)                          : (2*num_buses)             ]
eta     = Gamma_hat[(2*num_buses + 1)            : (2*num_buses + num_ReLU)  , (2*num_buses + 1)            : (2*num_buses + num_ReLU)  ]
B       = Gamma_hat[(2*num_buses + num_ReLU + 1) : (2*num_buses + 2*num_ReLU), (2*num_buses + num_ReLU + 1) : (2*num_buses + 2*num_ReLU)] 
epsilon = Gamma_hat[(2)                          : (2*num_buses)             , (2*num_buses + 1)            : (2*num_buses + num_ReLU)  ]
gamma   = Gamma_hat[(2)                          : (2*num_buses)             , (2*num_buses + num_ReLU + 1) : (2*num_buses + 2*num_ReLU)] 
delta   = Gamma_hat[(2*num_buses + 1)            : (2*num_buses + num_ReLU)  , (2*num_buses + num_ReLU + 1) : (2*num_buses + 2*num_ReLU)]
betan = value.(beta)


# %% test 2
diag(B) - betan

# test 3
v    = Gamma_hat[1,(2)                          : (2*num_buses)             ]
xnn    = Gamma_hat[1,(2*num_buses + 1)            : (2*num_buses + num_ReLU)  ]
#beta = Gamma_hat[1,(2*num_buses + num_ReLU + 1) : (2*num_buses + 2*num_ReLU)]

tt = zeros(200,1)
ij = 1
for ii = 1:nx
    tt[ij] = abs(delta[ii,ii] - xnn[ii])
    ij = ij + 1
    tt[ij] = abs(eta[ii,ii] -  (W0_bar[ii,:]'*epsilon[:,ii] + xnn[ii]*b0_bar[ii]))
    ij = ij + 1
end

# %%
Gamma = value.(u)*value.(u)'

Omega1 = A*Gamma*(A') - A*value.(u)*(b') - b*(A*value.(u))' + b*b';
Omega2 = (A*value.(u)-b)*((A*value.(u)-b)');

# %%

# %%
GAp = Gamma_num*(A')
Au  = A*u_num
kk  = 1
all_omega = zeros(200,1)

#ii_vec = zeros(length(all_constraints),1)
#jj_vec = zeros(length(all_constraints),1)

# %%

for ii = 1:size(A,1)
    for jj = 1:size(A,1)
          if ii <= jj
            if (ii,jj) in all_constraints
                      Omgea_ij = 0
                      Omgea_ij = Omgea_ij + b[ii]*b[jj]
                      Omgea_ij = Omgea_ij -b[ii]*Au[jj]
                      Omgea_ij = Omgea_ij -Au[ii]*b[jj]
                      Omgea_ij = Omgea_ij + A[ii,:]'*GAp[:,jj]
                      all_omega[kk] = Omgea_ij

                      kk = kk + 1
            end
          end
    end
end


# Activate the environment
using Pkg
Pkg.activate(".")

# Load the packages
using JuMP
using HDF5
using Plots
using Mosek
using Gurobi
using Random
using Graphs
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
verification_index   = 2
verification_routine = "Squared_Voltage_Magnitude"

# assess constraints for feasible dual (bounded primal)
initial_constraints = necessary_strong_constraints(ac_power_flow_model, verification_routine, verification_index)

# %% tighten the SDP?
strong_SDP          = true
sparse_SDP          = false
num_new_constraints = 250
include("src/verification.jl")

# initial verification
Results, all_constraints, new_constraints, verification_model = strong_SDP_verification(ac_power_flow_model, ac_nn_model, verification_routine, verification_index, initial_constraints, num_new_constraints, strong_SDP, sparse_SDP);

# %
#tt = copy([all_constraints; new_constraints])
#Results, all_constraints, new_constraints, verification_model = strong_SDP_verification(ac_power_flow_model, ac_nn_model, verification_routine, verification_index, tt, num_new_constraints, strong_SDP, sparse_SDP);

# %% 
strong_SDP          = true
sparse_SDP          = true
include("src/verification.jl")

Results_SP, all_constraints, new_constraints, verification_model = strong_SDP_verification(ac_power_flow_model, ac_nn_model, verification_routine, verification_index, initial_constraints, num_new_constraints, strong_SDP, sparse_SDP);

# %%
total_n = 0

for ii in 1:length(clique_groups)
    total_n = total_n + length(clique_groups[ii])^2
end

# Results_SP, all_constraints, new_constraints, verification_model = strong_SDP_verification(ac_power_flow_model, ac_nn_model, verification_routine, verification_index, tt, num_new_constraints, strong_SDP, sparse_SDP);

# %%

cadj, p, cadj_preperm = chordal_extension(Results_SP[:adj])


# %%
adj = Results_SP[:adj]
nb  = size(adj, 1)
diag_el = sum(adj, dims=1)[:]
W = Hermitian(-adj + spdiagm(0 => diag_el .+ 1))

F = cholesky(W)
L = sparse(F.L)
p = F.p
q = invperm(p)

Rchol = L - spdiagm(0 => diag(L))
f_idx, t_idx, V = findnz(Rchol)
cadj_preperm = sparse([f_idx;t_idx], [t_idx;f_idx], ones(2*length(f_idx)), nb, nb)
cadj         = cadj_preperm[q, q] # revert to original bus ordering (invert cholfact permutation)

# %%
clique_groups    = PowerModels._maximal_cliques(cadj, p)


# %%
# using MAT
#file = matopen("matfile.mat", "w")
#write(file, "Madj", Madj)
#close(file)

# %%
adj = Results_SP[:adj]
nb = size(adj, 1)
diag_el = sum(adj, dims=1)[:]
W = Hermitian(-adj + spdiagm(0 => diag_el .+ 1))
sparsity_pattern = abs.(W)

file = matopen("sparsity_pattern.mat", "w")
write(file, "sparsity_pattern", sparsity_pattern)
close(file)


# %% Test -- do known inds do what they are supposed to?
GH = Results[:Gamma_hat]
GH_test = 0*GH

cad = Results_SP[:chordal_adj]
for ii in 1:size(cad,1)
    cad[ii,ii] = 1
end

GH_test[cad.==1] = GH[cad.==1]

Gam_true = Results[:Gamma]
Gam_zero = GH_test[2:end,2:end]

csnt_true = Results[:A]*Gam_true*(Results[:A]') - Results[:A]*Results[:u]*(Results[:b]') - Results[:b]*(Results[:A]*Results[:u])' + Results[:b]*Results[:b]'
csnt_zero = Results[:A]*Gam_zero*(Results[:A]') - Results[:A]*Results[:u]*(Results[:b]') - Results[:b]*(Results[:A]*Results[:u])' + Results[:b]*Results[:b]'

# %%
ee = zeros(length(tt),1)
for ii in 1:length(tt)
    #println("----------")
    ee[ii] = csnt_true[tt[ii][1],tt[ii][2]] - csnt_zero[tt[ii][1],tt[ii][2]]
    #println(csnt_zero[tt[ii][1],tt[ii][2]])
    #println("----------")
end




# %%


for ii in 1:size(GH,1)
    GH_test[ii,ii] = GH[ii,ii]
end

for jj in 1:size(Results_SP[:knowninds],1)
    GH_test[Results_SP[:knowninds][jj,1],Results_SP[:knowninds][jj,2]] = GH[Results_SP[:knowninds][jj,1],Results_SP[:knowninds][jj,2]]
    GH_test[Results_SP[:knowninds][jj,2],Results_SP[:knowninds][jj,1]] = GH[Results_SP[:knowninds][jj,1],Results_SP[:knowninds][jj,2]]
end


# %% -- Test linking :)
Gamma_hat_TEST = [1            Results[:u]'
                  Results[:u]  Results_SP[:Gamma]]

clique_groups = Results_SP[:clique_groups]
pair_matrix(clique_groups) = [(i, j) for i in clique_groups, j in clique_groups]
#MM = zeros(length(clique_groups[1],length(clique_groups[1])
#clique_mats = Dict{Integer, Matrix}(1 => ones(length(clique_groups[1]),length(clique_groups[1])))
for ii in 1:size(clique_groups,1)
    numbus_group    = length(clique_groups[ii])
    MM = zeros(length(clique_groups[ii]),length(clique_groups[ii]))
    clique_mat_inds = pair_matrix(clique_groups[ii])
    for jj = 1:size(clique_mat_inds,1)
          for kk = 1:size(clique_mat_inds,2)
                MM[jj,kk] = Gamma_hat_TEST[clique_mat_inds[jj,kk][1],clique_mat_inds[jj,kk][2]]
          end
    end
    if ii == 1
        clique_mats = Dict{Integer, Matrix}(1 => MM)
    else
        clique_mats[ii] = MM
    end
end

# %% Populate


#for ii in 1:length(clique_groups)
 #   clique_mat_inds = pair_matrix(clique_groups[ii])
  #  for jj = 1:size(clique_mat_inds,1)
   #       for kk = 1:size(clique_mat_inds,2)
    #        #clique_mats[ii][jj][kk] = 
     #       println(Gamma_hat_TEST[clique_mat_inds[jj,kk][1],clique_mat_inds[jj,kk][2]])
      #    end
  #  end
#end

# %%

# 2. Link the groups!

clique_groups = Results_SP[:clique_groups]

tree = PowerModels._prim(PowerModels._overlap_graph(clique_groups))
overlapping_pairs = [Tuple(CartesianIndices(tree)[i]) for i in (LinearIndices(tree))[findall(x->x!=0, tree)]]
pair_matrix(clique_groups) = [(i, j) for i in clique_groups, j in clique_groups]
for (i, j) in [overlapping_pairs[1]]
    gi,    gj    = clique_groups[i], clique_groups[j]
    var_i, var_j = clique_mats[i],   clique_mats[j]
    Gi,    Gj    = pair_matrix(gi),  pair_matrix(gj)
    overlap_i, overlap_j = PowerModels._overlap_indices(Gi, Gj)
    indices = zip(overlap_i, overlap_j)
    for (idx_i, idx_j) in indices
        println("---------------")
        println(idx_i)
        println(idx_j)
        println("---------------")
    end
end


# %%

csnt = Results_SP[:A]*Results_SP[:Gamma]*(Results_SP[:A]') - Results_SP[:A]*Results_SP[:u]*(Results_SP[:b]') - Results_SP[:b]*(Results_SP[:A]*Results_SP[:u])' + Results_SP[:b]*Results_SP[:b]'



# %%
A = Results[:A]

ij = 1
jk = 2

A1 = A[:,ij]
A2 = A[:,jk]

(row_inds,col_inds,~) = findnz(A1[:]*(A2[:]'))


# %%
spy(Results_SP[:chordal_adj])

adj_test = Results_SP[:chordal_adj]

adj_test_out, p = chordal_extension(adj_test)

# %%
mc = _maximal_cliques(adj_test_out, p)

# %%
mcp = _mcs(adj_test_out)

# %%


function chordal_extension(adj::SparseMatrixCSC)
    nb = size(adj, 1)
    diag_el = sum(adj, dims=1)[:]
    W = Hermitian(-adj + spdiagm(0 => diag_el .+ 1))

    F = cholesky(W)
    L = sparse(F.L)
    p = F.p
    q = invperm(p)

    Rchol = L - spdiagm(0 => diag(L))
    f_idx, t_idx, V = findnz(Rchol)
    cadj = sparse([f_idx;t_idx], [t_idx;f_idx], ones(2*length(f_idx)), nb, nb)
    cadj = cadj[q, q] # revert to original bus ordering (invert cholfact permutation)
    
    return cadj, p
end


function _maximal_cliques(cadj::SparseMatrixCSC, peo::Vector{Int})
    nb = size(cadj, 1)

    # use peo to obtain one clique for each vertex
    cliques = Vector(undef, nb)
    for (i, v) in enumerate(peo)
        Nv = findall(x->x!=0, cadj[:, v])
        cliques[i] = union(v, intersect(Nv, peo[i+1:end]))
    end

    # now remove cliques that are strict subsets of other cliques
    mc = Vector()
    for c1 in cliques
        # declare clique maximal if it is a subset only of itself
        if sum([issubset(c1, c2) for c2 in cliques]) == 1
            push!(mc, c1)
        end
    end
    # sort node labels within each clique
    mc = [sort(c) for c in mc]
    return mc
end

function _mcs(A)
    n = size(A, 1)
    w = zeros(Int, n)
    peo = zeros(Int, n)
    unnumbered = collect(1:n)

    for i = n:-1:1
        z = unnumbered[argmax(w[unnumbered])]
        filter!(x -> x != z, unnumbered)
        peo[i] = z

        Nz = findall(x->x!=0, A[:, z])
        for y in intersect(Nz, unnumbered)
            w[y] += 1
        end
    end
    return peo
end

# %%
pglib_case_name        = "pglib_opf_case200_activ_cost.m"
powerflow_data_file    = data_folder*"case200_data.h5"
powerflow_network_file = data_folder*"case200_network.h5"
nn_parameter_file      = model_folder*"case200_neural_network.h5"


ac_nn_model = build_nn_model(nn_parameter_file)


# %% ==================
case = 200

pglib_case_name        = "pglib_opf_case200_activ_cost.m"
powerflow_data_file    = data_folder*"case200_data.h5"
powerflow_network_file = data_folder*"case200_network.h5"
nn_parameter_file      = model_folder*"case200_neural_network.h5"
num_runs               = 4
num_new_constraints    = 300

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
ii = 4
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
    :mosek_tols                => 1e-8,
    :num_new_constraints       => num_new_constraints,
    :iteration                 => 0,
    :solutions                 => Dict())

# run stt
sttlog1 = stt_algorithm(num_runs, ac_power_flow_model, ac_nn_model, verification_params)

verification_params[:mosek_tols] = 1e-7
sttlog2 = stt_algorithm(num_runs, ac_power_flow_model, ac_nn_model, verification_params)

verification_params[:mosek_tols] = 1e-6
sttlog3 = stt_algorithm(num_runs, ac_power_flow_model, ac_nn_model, verification_params)


# run miqp
#timelim  = sttlog1[:timelog][end]
#miqplog1 = miqp_verification(ac_power_flow_model, ac_nn_model, verification_params, timelim)

# %%
using SparseArrays
using LinearAlgebra
A = sprandn(10,10, 0.01)
#A = randn(10,10)
A = A + A' + 10*I
C = cholesky(A)

# legacy plots
#plot!(miqplog[:timelog], miqplog[:boundlog], yaxis = :log, xaxis = :log10, linewidth=4, opacity=.75, color = redd, label = "MIQP iteration", foreground_color_legend = nothing) #legend = (position = :bottom, titleposition = :left)) #foreground_color_legend = nothing)
#timevec, boundvec, eig12vec, eig13vec = sdp_data(verification_params)
#plot(miqplog[:timelog],miqplog[:boundlog])
#plot!(timevec, boundvec, yaxis = :log, xaxis = :log)