# Train power flow model (Vr, Vi) -> (V^2, P, Q, Ift^2, Itf^2)
function train_powerflow_model(network_data::Dict,powerflow_data_file::String,powerflow_network_file::String,BSON_file::String,hyperparams::Dict)
    
    # Parse hyperparameters
    num_ReLU                = hyperparams[:num_ReLU]
    dense_epoch_iterations  = hyperparams[:dense_epoch_iterations]
    adam                    = hyperparams[:adam]
    dense_learning_rate     = hyperparams[:dense_learning_rate]
    decay_tup               = hyperparams[:decay_tup]
    print_val               = hyperparams[:print_val]
    load_model              = hyperparams[:load_model]
    batch_size              = hyperparams[:batch_size]
    shuffle_data            = hyperparams[:shuffle_data]
    percent_sparse          = hyperparams[:percent_sparse]
    dont_scale_std          = hyperparams[:dont_scale_std]
    setzero_threshold       = hyperparams[:setzero_threshold]

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
    end

    # Load newtork information
    c = h5open(powerflow_network_file, "r") do file
        global ref_bus      = read(file, "ref_bus") 
        global num_buses    = read(file, "num_buses") 
        global num_lines    = read(file, "num_lines") 
    end

    # Build Optimal Linearization Planes
    #    ASSUME: The Linearization Planes come from the first column
    #    of the data structures -- this is their equilibrium
    Yb, NY, NYft, NYtf, E1, E2, E, YSftE1, YStfE2, NYImft, NYImtf, YLft, YLtf, YlDC = network_structures(network_data)
    Vm    = Vm_TrDat[:,1]
    Theta = Theta_TrDat[:,1]
    Vc    = Vm.*exp.(Theta*im)
    Vr    = real(Vc)
    Vi    = imag(Vc)
    Ic    = Yb*Vc

    # Jacobians (Vr,Vi) -> output
    Polar = false
    JV2   = Jacobian_VrVi_to_V2(Vc)
    JPQ   = Jacobian_VrVi_to_PQ(Theta,Vc,Ic,NY,Polar)
    JI2ft = Jacobian_VrVi_to_I2(Vc,YLft,YSftE1,NYImft)
    JI2tf = Jacobian_VrVi_to_I2(Vc,YLtf,YStfE2,NYImtf)
    J0    = [JV2; JPQ; JI2ft; JI2tf];
    J0    = J0[:,Not(num_buses+ref_bus)] # delete the reference bus column

    # Residual
    x0   = [Vr; Vi[Not(ref_bus)]];
    f_x0 = [Vm.^2
            real(Vc.*conj(Yb*Vc));
            imag(Vc.*conj(Yb*Vc));
            abs.((YLft+YSftE1)*Vc).^2;
            abs.((YLtf+YStfE2)*Vc).^2]
    r0 = f_x0 - J0*x0

    # Set any small values to 0 -- this is fine, since you will train around this point anyways
    r0[abs.(r0) .< setzero_threshold] .= 0
    J0[abs.(J0) .< setzero_threshold] .= 0

    # Define Flux Training Model & Train
    input  = [Vr_TrDat; Vi_TrDat[Not(ref_bus),:]];
    output = [Vm_TrDat.^2; Pinj_TrDat; Qinj_TrDat; Ift2_TrDat; Itf2_TrDat];

    # Subtract the feedthrough term
    output_residual = output - (J0*input .+ r0)

    # Statistics
    In_Mean  = mean(input,dims = 2)
    In_STD   = std(input,dims = 2)
    Out_Mean = mean(output_residual,dims = 2)
    Out_STD  = std(output_residual,dims = 2)

    # If the STD is too small, just set it to 1 -- this can happen at a node that
    # always has a zero load! Do this to the mean as well
    In_STD[In_STD     .< setzero_threshold/100] .= 1.0
    Out_STD[Out_STD   .< setzero_threshold/100] .= 1.0
    In_Mean[In_Mean   .< setzero_threshold/100] .= 0.0
    Out_Mean[Out_Mean .< setzero_threshold/100] .= 0.0

    if dont_scale_std == true
        In_STD  .= 1
        Out_STD .= 1
    end

    # Normalize!
    input_normal  = (input .- In_Mean)./In_STD
    output_normal = (output_residual .- Out_Mean)./Out_STD

    # Define Input/Output Sizes
    num_in  = size(input_normal,1)
    num_out = size(output_normal,1)

    # Build Model: 2 Layers (ReLU & Linear)
    # **************************************** #
    #=  ############### This code took forever. Don't delete. ###############
    rₛ₂        = Int(rₛ/2)
    layer_1    = Dense(nᵢ,rₛ₂,relu)
    SC         = SkipConnection(layer_1, (mx, x) -> cat(mx, x, dims=1))
    layer_2    = Dense(nᵢ+rₛ₂,rₛ₂,relu)
    ch         = Chain(SC,layer_2)
    pl         = Parallel(vcat,layer_1,ch)
    layer_3    = Dense(rₛ,nₒ,identity,bias = outer_bias)
    LRPF_model = Chain(pl,layer_3)
    # **************************************** #
    =#

    # Single Layer
    NN_model = Chain(
        Dense(num_in,num_ReLU,relu),
        Dense(num_ReLU,num_out,identity))

    # Load model?
    if load_model
        @load BSON_file ModelParams
        Flux.loadparams!(NN_model, ModelParams)
    end

    # Build Loss Function
    loss(x,y)    = Flux.Losses.mse(NN_model(x), y)
    initial_loss = loss(input_normal, output_normal)
    println("Starting Loss: $initial_loss")

    # Prepare the validation data
    input_val           = [Vr_ValDat; Vi_ValDat[Not(ref_bus),:]];
    output_val          = [Vm_ValDat.^2; Pinj_ValDat; Qinj_ValDat; Ift2_ValDat; Itf2_ValDat];
    output_residual_val = output_val - (J0*input_val .+ r0)
    input_normal_val    = (input_val .- In_Mean)./In_STD
    output_normal_val   = (output_residual_val .- Out_Mean)./Out_STD

    # Loop and Train
    if adam
        opt = Flux.Optimise.ADAM(dense_learning_rate, decay_tup)
    else
        opt = Descent(dense_learning_rate)
    end

    # Loop over dataset
    data   = Flux.Data.DataLoader((input_normal, output_normal),batchsize=batch_size,shuffle=shuffle_data)
    params = Flux.params(NN_model)

    # Track loss metrics
    losses        = []
    val_losses    = []
    best_NN_model = []
    best_validation_loss = 1e4

    # Loop [Alternative to looping: @epochs 100 Flux.train!(loss, params, data, opt);]
    for ii in 1:dense_epoch_iterations
        Flux.train!(loss, params, data, opt)
        # Alternatively: Flux.train!(loss, Flux.params(NN_model), [(x, y)], opt)

        training_loss   = loss(input_normal, output_normal)
        validation_loss = loss(input_normal_val, output_normal_val)
        push!(losses, training_loss)
        push!(val_losses, validation_loss)

        # Print
        if ii % print_val == 0
            println("$ii Training Loss: $training_loss")
            print("\n")
        end

        # Continually track the best validation error, and save the associated model
        if validation_loss < best_validation_loss
            best_validation_loss = copy(validation_loss)
            best_NN_model        = deepcopy(NN_model)
        end
    end

    # Now, plot the training and validation loss
    plot()
    display(plot(losses, yaxis = log, xlabel = "training epoch", ylabel="loss", label = "training data", title = "Has validation loss saturated?"))
    display(plot!(val_losses, yaxis = log, xlabel = "training epoch", ylabel="loss", label = "validation data"))

    # Return
    return Dict(:nn => best_NN_model, :losses => losses, :val_losses => val_losses, 
                :input => input, :output => output, :input_normal => input_normal, 
                :output_normal => output_normal, :input_normal_val => input_normal_val, 
                :output_normal_val => output_normal_val, :In_Mean => In_Mean, :In_STD => In_STD,
                :Out_Mean => Out_Mean, :Out_STD =>  Out_STD, :J0 => J0, :r0 => r0)
end

# %% Train PF Solver - Sparse
function train_powerflow_model_sparse(dense_model_data::Dict, hyperparams::Dict)
    
    # Parse hyperparameters
    num_ReLU                = hyperparams[:num_ReLU]
    sparse_epoch_iterations = hyperparams[:sparse_epoch_iterations]
    adam                    = hyperparams[:adam]
    sparse_learning_rate    = hyperparams[:sparse_learning_rate]
    decay_tup               = hyperparams[:decay_tup]
    print_val               = hyperparams[:print_val]
    load_model              = hyperparams[:load_model]
    batch_size              = hyperparams[:batch_size]
    shuffle_data            = hyperparams[:shuffle_data]
    percent_sparse          = hyperparams[:percent_sparse]
    setzero_threshold       = hyperparams[:setzero_threshold]

    # Parse
    NN_model          = dense_model_data[:nn]
    input_normal      = dense_model_data[:input_normal]
    output_normal     = dense_model_data[:output_normal]
    input_normal_val  = dense_model_data[:input_normal_val]
    output_normal_val = dense_model_data[:output_normal_val]

    # Sparsify!
    W0 = deepcopy(NN_model[1].weight)
    W1 = deepcopy(NN_model[2].weight)

    # Test
    numel_W0   = length(W0)
    numW0_zero = Int(round(percent_sparse*numel_W0))
    numel_W1   = length(W1)
    numW1_zero = Int(round(percent_sparse*numel_W1))

    # Which elements do we set to 0?
    W0_sort_inds  = sortperm(vec(abs.(W0)))
    W1_sort_inds  = sortperm(vec(abs.(W1)))
    sparse_inds_W0 = W0_sort_inds[1:numW0_zero]
    sparse_inds_W1 = W1_sort_inds[1:numW1_zero]

    # Sparsify!
    W0[sparse_inds_W0] .= 0
    W1[sparse_inds_W1] .= 0

    # Update the NN
    NN_model[1].weight .= W0
    NN_model[2].weight .= W1

    # Define Input/Output Sizes
    num_input  = size(input_normal,1)
    num_output = size(output_normal,1)

    # Build Loss Function
    loss(x,y) = Flux.Losses.mse(NN_model(x), y)
    init_loss = loss(input_normal, output_normal)
    println("Starting Loss: $init_loss")

    ## Loop and Train
    if adam
        opt = Flux.Optimise.ADAM(sparse_learning_rate, decay_tup)
    else
        opt = Descent(sparse_learning_rate)
    end

    # Loop over dataset
    data   = Flux.Data.DataLoader((input_normal, output_normal),batchsize=batch_size,shuffle=shuffle_data)
    params = Flux.params(NN_model)

    # Track loss metrics
    losses        = []
    val_losses    = []
    best_sparse_NN_model = []
    best_validation_loss = 1e4

    # Loop [Alternative to looping: @epochs 100 Flux.train!(loss, pm, data, opt);]
    for ii in 1:sparse_epoch_iterations
        for (x, y) in data
          grad = gradient(params) do
              loss(x, y)
          end
          grad[NN_model[1].weight][sparse_inds_W0] .= 0
          grad[NN_model[2].weight][sparse_inds_W1] .= 0
          Flux.update!(opt, params, grad)
        end

        training_loss   = loss(input_normal, output_normal)
        validation_loss = loss(input_normal_val, output_normal_val)
        push!(losses, training_loss)
        push!(val_losses, validation_loss)

        # Print
        if ii % print_val == 0
            println("$ii Loss: $training_loss")
             print("\n")
        end

        # Continually track the best validation error, and save the associated model
        if validation_loss < best_validation_loss
            best_validation_loss = copy(validation_loss)
            best_sparse_NN_model = deepcopy(NN_model)
        end
    end

    # drive the smallest values to 0!
    n1  = nnz(sparse(best_sparse_NN_model[1].weight))
    n2  = nnz(sparse(best_sparse_NN_model[2].weight))
    n3  = nnz(sparse(best_sparse_NN_model[1].bias))
    n4  = nnz(sparse(best_sparse_NN_model[2].bias))

    # actually do it
    best_sparse_NN_model[1].weight[abs.(best_sparse_NN_model[1].weight) .< setzero_threshold] .= 0
    best_sparse_NN_model[2].weight[abs.(best_sparse_NN_model[2].weight) .< setzero_threshold] .= 0
    best_sparse_NN_model[1].bias[abs.(best_sparse_NN_model[1].bias)     .< setzero_threshold] .= 0
    best_sparse_NN_model[2].bias[abs.(best_sparse_NN_model[2].bias)     .< setzero_threshold] .= 0

    # update the count
    n1sp  = nnz(sparse(best_sparse_NN_model[1].weight))
    n2sp  = nnz(sparse(best_sparse_NN_model[2].weight))
    n3sp  = nnz(sparse(best_sparse_NN_model[1].bias))
    n4sp  = nnz(sparse(best_sparse_NN_model[2].bias))

    # print
    percent_zero = 100*((n1+n2+n3+n4) - (n1sp+n2sp+n3sp+n4sp))/(n1+n2+n3+n4)
    @info("New model parameters set to zero (after sparse training): $percent_zero%")

    # How did this change the loss?
    new_validation_loss     = Flux.Losses.mse(best_sparse_NN_model(input_normal_val), output_normal_val)
    accuracy_loss_threshold = 100*(1 - best_validation_loss/new_validation_loss)
    @info("Initial loss: $best_validation_loss. New loss: $new_validation_loss.")
    @info("Loss in accuracy: $accuracy_loss_threshold%.")

    plot()
    display(plot(losses, yaxis = log, xlabel = "training epoch", ylabel="loss", label = "training data", title = "Has validation loss saturated?"))
    display(plot!(val_losses, yaxis = log, xlabel = "training epoch", ylabel="loss", label = "validation data"))

    # Return Values
    return Dict(:nn => best_sparse_NN_model, :losses => losses, :val_losses => val_losses, :accuracy_loss_threshold => accuracy_loss_threshold, :setzero_threshold => setzero_threshold)
end

# Save NN to file
function save_nn(sparse_model::Dict, dense_model_data::Dict, nn_parameter_file::String, BSON_file::String)

    # parse the NN itself
    best_sparse_NN_model    = sparse_model[:nn]
    setzero_threshold       = sparse_model[:setzero_threshold]
    accuracy_loss_threshold = sparse_model[:accuracy_loss_threshold]

    # parse the rest
    In_Mean  = dense_model_data[:In_Mean]
    In_STD   = dense_model_data[:In_STD]
    Out_Mean = dense_model_data[:Out_Mean]
    Out_STD  = dense_model_data[:Out_STD]
    J0       = dense_model_data[:J0]
    r0       = dense_model_data[:r0]

    # save the NN BSON File
    model_params = Flux.params(best_sparse_NN_model);
    @save BSON_file model_params

    # peel parameters
    W0 = best_sparse_NN_model[1].weight
    b0 = best_sparse_NN_model[1].bias
    W1 = best_sparse_NN_model[2].weight
    b1 = best_sparse_NN_model[2].bias

    # Convert 
    In_Mean  = convert(Matrix{Float64}, In_Mean)
    In_STD   = convert(Matrix{Float64}, In_STD)
    Out_Mean = convert(Matrix{Float64}, Out_Mean)
    Out_STD  = convert(Matrix{Float64}, Out_STD)
    J0       = convert(Matrix{Float64}, J0)
    r0       = convert(Vector{Float64}, r0)

    # we don't have big-M values yet
    M_max = +1000.0*ones(length(b0))
    M_min = -1000.0*ones(length(b0))

    # Save additional NN Data
    fid = h5open(nn_parameter_file, "w") do file
          write(file, "In_Mean", In_Mean)
          write(file, "In_STD", In_STD)
          write(file, "Out_Mean", Out_Mean)
          write(file, "Out_STD", Out_STD)
          write(file, "J0", J0)
          write(file, "r0", r0)
          write(file, "W0", W0)
          write(file, "b0", b0)
          write(file, "W1", W1)
          write(file, "b1", b1)
          write(file, "M_max", M_max)
          write(file, "M_min", M_min)
          write(file, "setzero_threshold", setzero_threshold)
          write(file, "accuracy_loss_threshold", accuracy_loss_threshold)
    end
end

# Save NN to file
function save_nn_with_bigM(ac_nn_model::ac_nn_model_instance, nn_parameter_file::String)

    # parse
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

    # thresholding
    setzero_threshold       = ac_nn_model.setzero_threshold
    accuracy_loss_threshold = ac_nn_model.accuracy_loss_threshold

    # save nn
    fid = h5open(nn_parameter_file, "w") do file
          write(file, "In_Mean", In_Mean)
          write(file, "In_STD", In_STD)
          write(file, "Out_Mean", Out_Mean)
          write(file, "Out_STD", Out_STD)
          write(file, "J0", J0)
          write(file, "r0", r0)
          write(file, "W0", W0)
          write(file, "b0", b0)
          write(file, "W1", W1)
          write(file, "b1", b1)
          write(file, "M_max", M_max)
          write(file, "M_min", M_min)
          write(file, "setzero_threshold", setzero_threshold)
          write(file, "accuracy_loss_threshold", accuracy_loss_threshold)
    end
end

function build_nn_model(nn_parameter_file::String)
    # Load NN parameter data
    c = h5open(nn_parameter_file, "r") do file
        global In_Mean  = read(file, "In_Mean")
        global In_STD   = read(file, "In_STD")
        global Out_Mean = read(file, "Out_Mean")
        global Out_STD  = read(file, "Out_STD")
        global J0       = read(file, "J0")
        global r0       = read(file, "r0")
        global W0       = read(file, "W0")
        global b0       = read(file, "b0")
        global W1       = read(file, "W1")
        global b1       = read(file, "b1")
        global M_max    = read(file, "M_max")
        global M_min    = read(file, "M_min")
        global setzero_threshold       = read(file, "setzero_threshold")
        global accuracy_loss_threshold = read(file, "accuracy_loss_threshold")
    end

    # build ac_nn_model
    ac_nn_model = ac_nn_model_instance(
        W0,
        b0,
        W1,
        b1,
        M_max,
        M_min,
        In_Mean,
        In_STD,
        Out_Mean,
        Out_STD,
        J0,
        r0,
        setzero_threshold,
        accuracy_loss_threshold)

    # Output
    return ac_nn_model
end

# Plot NN results
function plot_nn_results(dense_model_data::Dict,sparse_model::Dict,sample::Int)

    # Call the NN data and parameters itself
    r0               = dense_model_data[:r0]
    J0               = dense_model_data[:J0]
    input            = dense_model_data[:input]
    input_normal     = dense_model_data[:input_normal]
    Out_Mean         = dense_model_data[:Out_Mean]
    Out_STD          = dense_model_data[:Out_STD]
    output           = dense_model_data[:output]
    num_buses        = Int((size(input,1)+1)/2)

    # call the nn itslef
    nn = sparse_model[:nn]

    # nn prediction
    predicted_output = (Out_STD.*nn(input_normal) .+ Out_Mean) + (J0*input .+ r0);

    # subplots
    l  = @layout [a b; c d]
    p1 = plot(predicted_output[1:num_buses,sample],legend = false) # Plot V^2
    p1 = plot!(output[1:num_buses,sample], title="V^2",legend = false)
    p2 = plot(predicted_output[num_buses+1:2*num_buses,sample],legend = false) # Plot P
    p2 = plot!(output[num_buses+1:2*num_buses,sample], title="P",legend = false)
    p3 = plot(predicted_output[2*num_buses+1:3*num_buses,sample],legend = false) # Plot Q
    p3 = plot!(output[2*num_buses+1:3*num_buses,sample], title="Q",legend = false)
    p4 = plot(predicted_output[3*num_buses+1:end,sample],legend = false) # Plot I^2
    p4 = plot!(output[3*num_buses+1:end,sample], title="I^2",legend = false)
    plot(p1, p2, p3, p4, layout = l)
end

# run big-M
function run_big_M(network_data::Dict, powerflow_network_file::String, nn_parameter_file::String, hyperparams::Dict)

    # build "ac_power_flow_model"
    ac_power_flow_model = build_acpf_model(network_data, powerflow_network_file)

    # build "nn_model"
    ac_nn_model = build_nn_model(nn_parameter_file)

    # compute big-M
    mip_gap     = hyperparams[:mip_gap]
    time_lim    = hyperparams[:time_lim]
    ac_nn_model = compute_big_M(ac_power_flow_model, ac_nn_model, mip_gap, time_lim)

    # re-save the model
    save_nn_with_bigM(ac_nn_model, nn_parameter_file)

    return ac_nn_model
end

# compute big-M
function compute_big_M(ac_power_flow_model::ac_power_flow_model_instance,ac_nn_model::ac_nn_model_instance,mip_gap::Float64, time_lim::Float64)
    # step 1: interval arithmetic to estimate big-M
    ac_nn_model   = compute_bigM_IntervalArithmetic(ac_power_flow_model,ac_nn_model)

    # step 2: solve LP to tighten big-M
    LP_relaxation = true
    ac_nn_model   = compute_bigM_MILP(ac_power_flow_model,ac_nn_model,mip_gap,time_lim,LP_relaxation)

    # step 3: solve MILP to find the very best big-M
    LP_relaxation = false
    ac_nn_model   = compute_bigM_MILP(ac_power_flow_model,ac_nn_model,mip_gap,time_lim,LP_relaxation)
end


# compute big-M
function compute_bigM_IntervalArithmetic(ac_power_flow_model::ac_power_flow_model_instance,ac_nn_model::ac_nn_model_instance)

    # Build model
    model = Model(Gurobi.Optimizer)
    empty!(model)

    # Model Settings
    set_optimizer_attribute(model, "OutputFlag", 0)

    # Big-M vectors
    M_max_vector  = Float64[]
    M_min_vector  = Float64[]

    # Call nn Parameters
    W0       = ac_nn_model.W0
    b0       = ac_nn_model.b0
    In_Mean  = ac_nn_model.In_Mean
    In_STD   = ac_nn_model.In_STD

    # Call power flow constraints
    V_max    = ac_power_flow_model.V_max
    V_min    = ac_power_flow_model.V_min

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

    nn_input        = [vr; vi]
    nn_input_normal = (nn_input - In_Mean)./In_STD
    x_ReLU_in       = W0*nn_input_normal + b0

    # =============================================================== #
    # core question: what is the maximum possible value of "x_ReLU_in"?

    # Build objective function
    for ii = 1:num_ReLU
          println(ii)
          println("Solving maximum")

          # Just grab the ii-th value of the sigma-input
          obj = x_ReLU_in[ii] ##########################

          # First, maximize
          @objective(model, Max, obj)
          set_optimizer_attribute(model, "OutputFlag", 0)
          optimize!(model)
          if (Int(termination_status(model)) == 1) && (Int(primal_status(model)) == 1)
                push!(M_max_vector,value(obj))
          else
                println("***Something is wrong with the solution***")
          end

          # Second, minimize
          println("Solving minumum")
          @objective(model, Min, obj)
          set_optimizer_attribute(model, "OutputFlag", 0)
          optimize!(model)
          if (Int(termination_status(model)) == 1) && (Int(primal_status(model)) == 1)
                push!(M_min_vector,value(obj))
          else
                println("***Something is wrong with the solution***")
          end
    end

    # Now, update 
    ac_nn_model.M_max = M_max_vector
    ac_nn_model.M_min = M_min_vector

    # Output
    return ac_nn_model
end

# compute big-M
function compute_bigM_MILP(ac_power_flow_model::ac_power_flow_model_instance,ac_nn_model::ac_nn_model_instance,mip_gap::Float64,time_lim::Float64,LP_relaxation::Bool)

    # Build model
    model = Model(Gurobi.Optimizer)
    empty!(model)

    # Big-M vectors
    M_max_vector       = Float64[] # this is what we have to use, to be safe (it is the bound)
    M_min_vector       = Float64[] # this is what we have to use, to be safe (it is the bound)
    M_max_vector_best  = Float64[] # this is what Gurobi found
    M_min_vector_best  = Float64[] # this is what Gurobi found

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
    Ift2_min = -5e-2*ones(length(Ift2_max),1) # previously: 0*Ift2_max --- -1e-2 for numerics
    Itf2_min = -5e-2*ones(length(Ift2_max),1) # previously: 0*Itf2_max --- -1e-2 for numerics

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

    if LP_relaxation == true
          # This is the LP relaxation!
          @variable(model, 0 <= beta[1:num_ReLU] <= 1)      # linear variable
          @variable(model, 0 <= x_ReLU_out[1:num_ReLU])     # ReLU output
          set_optimizer_attribute(model, "OutputFlag", 0)
    else
          # This is the true MILP
          @variable(model, beta[1:num_ReLU], binary = true)      # Binary variable
          @variable(model, 0 <= x_ReLU_out[1:num_ReLU])          # ReLU output

          # set gap
          set_optimizer_attribute(model, "MIPGap", mip_gap)
          set_optimizer_attribute(model, "TimeLimit", time_lim)
          set_optimizer_attribute(model, "OutputFlag", 0)
    end

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
    
    # Build objective function
    for ii = 1:num_ReLU
          println(ii)
          println("Solving maximum")
          # Just grab the ii-th value of the sigma-input
          obj = x_ReLU_in[ii] ##########################

          # First, maximize
          @objective(model, Max, obj)
          #set_optimizer_attribute(model, "OutputFlag", 0)
          optimize!(model)
          if (Int(primal_status(model)) == 1)
            if (Int(termination_status(model)) == 1)
                # noting to say -- optimal solution found
            elseif (Int(termination_status(model)) == 12)
                println("***Time limit reached, but we have a feasible solution***")
            end
                push!(M_max_vector,     objective_bound(model))
                push!(M_max_vector_best,value(obj))
          else
                println("***Something is wrong with the solution***")
                println(termination_status(model))
          end

          # Second, minimize
          println("Solving minumum")
          @objective(model, Min, obj)
          #set_optimizer_attribute(model, "OutputFlag", 0)
          optimize!(model)
          
          if (Int(primal_status(model)) == 1)
            if (Int(termination_status(model)) == 1)
                # noting to say -- optimal solution found
            elseif (Int(termination_status(model)) == 12)
                println("***Time limit reached, but we have a feasible solution***")
            end
                push!(M_min_vector,     objective_bound(model))
                push!(M_min_vector_best,value(obj))
          else
                println("***Something is wrong with the solution***")
                println(termination_status(model))
          end
    end

    for ii = 1:length(M_max_vector)
        @info "Gap $ii:"
        println(((M_max_vector[ii]/M_max_vector_best[ii]) - 1)*100)
        println(((M_min_vector[ii]/M_min_vector_best[ii]) - 1)*100)
    end

    # Now, update 
    ac_nn_model.M_max      = M_max_vector
    ac_nn_model.M_min      = M_min_vector
    println(M_max_vector_best)
    println(M_min_vector_best)
    
    # Output
    return ac_nn_model
end