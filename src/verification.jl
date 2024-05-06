function strong_SDP_verification(ac_power_flow_model::ac_power_flow_model_instance, ac_nn_model::ac_nn_model_instance, verification_params::Dict)
      
      # ***Reset*** the verification dict
      verification_params[:initial_constraints]       = [0]
      verification_params[:active_constraints]        = [0]
      verification_params[:potential_new_constraints] = [0]
      verification_params[:voltage_constraints]       = [0]
      verification_params[:iteration]                 = 1
      verification_params[:solutions]                 = Dict()

      # assess constraints for feasible dual (bounded primal)
      verification_params = necessary_strong_constraints(ac_power_flow_model, verification_params)

      # Build model
      model = Model(Mosek.Optimizer)
      empty!(model)

      # Set tolerances
      mosek_tols = verification_params[:mosek_tols]
      set_optimizer_attribute(model, "QUIET", false)
      set_optimizer_attribute(model, "MSK_IPAR_LOG_CUT_SECOND_OPT",       0)
      set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS",   mosek_tols)
      set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS",   mosek_tols)
      set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", mosek_tols)
      set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_MU_RED",  mosek_tols)
      set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_INFEAS",  mosek_tols)

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
      M_max    = ac_nn_model.M_max
      M_min    = ac_nn_model.M_min
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
      Ift2_min = -5e-2*ones(length(Ift2_max),1) # previously: 0*Ift2_max --- -1e-2 for numerics
      Itf2_min = -5e-2*ones(length(Ift2_max),1) # previously: 0*Itf2_max --- -1e-2 for numerics

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

      # Define the A matrix and the b vector (append below)
      A = [Ab;
            A0nn;
            AJ Apf*W1_bar spzeros(2*no,nb)]
      b = [bb;
            b0nn;
            bpf-Apf*b1_bar]

      # Extra cuts :)
      nnM_max = ac_nn_model.M_max .+ 1e-6
      nnM_min = ac_nn_model.M_min .- 1e-6
      m_int = nnM_max./(nnM_max - nnM_min)
      b_int = -nnM_min.*m_int
      for ii in 1:nx
            if nnM_max[ii] < 0
                  m_int[ii] = 0
                  b_int[ii] = 0
            elseif nnM_min[ii] > 0
                  m_int[ii] = 1
                  b_int[ii] = 0
            end
      end
      Ix  = sparse(I,nx,nx)
      Zb  = spzeros(nb,nb)
      AAn = [-Diagonal(m_int)*W0_bar Ix Zb]
      bbn = b_int + m_int.*b0_bar

      # Define the A matrix and the b vector
      include_extra = 0
      if include_extra == 1
            A = [A; AAn]
            b = [b; bbn]
      end

      # sanitize
      num_params_below_threshold = sum(0 .< abs.(b) .< setzero_threshold) + sum(0 .< abs.(A) .< setzero_threshold)
      @info("Number of model parameters below $setzero_threshold: $num_params_below_threshold")

      # Strengthen the SDP formulation
      if sparse_SDP == true
            # grab the constraint indices
            num_verts = num_ReLU + num_ReLU + 2*num_buses
            knowninds = [1 2]

            # add the linear terms
            for ij in 3:num_verts
                  knowninds = vcat(knowninds,[1 ij])
            end

            # loop over additional constraints
            for ij in 1:(num_verts-1)           # don't include "1"
                  for jk in 1:(num_verts-1)     # don't include "1"
                        # grab the outer product
                        A1 = A[:,ij]
                        A2 = A[:,jk]

                        # the outer product is associated with X_{ij,jk}, from X = uu' (no 1!)
                        (row_inds,col_inds,~) = findnz(A1[:]*(A2[:]'))

                        # convert to tuple
                        nnz_inds = tuple.(row_inds,col_inds)

                        if !isempty(intersect(initial_constraints,nnz_inds))
                              # in this case, there is an intersection -- save the indices!
                              knowninds = vcat(knowninds,[ij+1 jk+1])
                        end
                  end
            end

            # next, add in the equality constraints
            # 1. diag(B) .== beta is implicitly captured, since diag(B) is on the main diagonal, and beta is linear
            # 2. diag(eta) is also implicitly captured
            # 3. delta and epsilon are not..
            # Indices reminder :::
            # #### nv = 2*num_buses - 1
            # #### nx = num_ReLU
            # #### nb = num_ReLU
            for ii = 1:nx
                  # for delta[ii,ii] == x[ii]
                  knowninds = vcat(knowninds,[(1+nv+ii) (1+nv+nx+ii)])
                  
                  # for      eta[ii,ii] ==  W0_bar[ii,:]'*epsilon[:,ii] + x[ii]*b0_bar[ii]
                  # simpler: W0_bar[ii,:]'*epsilon[:,ii]
                  for jj = 1:size(W0_bar,2)
                        if W0_bar[ii,jj] != 0
                              # include -- we need this value of epsilon
                              knowninds = vcat(knowninds,[(1+jj) (1+nv+ii)])
                        end
                  end
            end

            # using "knowninds", build the associated connected graph and find the cliques
            clique_groups, adj, chordal_adj = buildgraphadj(num_verts,knowninds)

            # Now, build a single matrix which has these variables in these spaces
            @variable(model, u_mat[1:(num_ReLU + num_ReLU + 2*num_buses - 1)])
            Gam_mat = Matrix{JuMP.AffExpr}(undef, (num_ReLU + num_ReLU + 2*num_buses - 1), (num_ReLU + num_ReLU + 2*num_buses - 1))

            # how many variables do we need?
            num_diag = num_ReLU + num_ReLU + 2*num_buses - 1
            num_vars = nnz(triu(chordal_adj[2:end,2:end])) + num_diag
            @variable(model, uum[1:num_vars])
            var_locs = triu(chordal_adj[2:end,2:end]) + sparse(I,num_diag,num_diag)
            Gam_mat[var_locs.==1] = uum

            # update the rest of the matrix and set all "#undef" values to 0
            for ii in 1:(num_ReLU + num_ReLU + 2*num_buses - 1)
                  for jj in 1:(num_ReLU + num_ReLU + 2*num_buses - 1)
                        if jj > ii
                              if isassigned(Gam_mat,ii,jj)
                                    # assign the other side too!
                                    Gam_mat[jj,ii] = Gam_mat[ii,jj]
                              else
                                    Gam_mat[jj,ii] = 0
                                    Gam_mat[ii,jj] = 0
                              end
                        end
                  end
            end

            # now, build the full matrix
            Gamma_hat = [1     u_mat';
                        u_mat Gam_mat]

            # Now, we just call each group, and we constrain it to be PSD
            pair_matrix(clique_groups) = [CartesianIndex(i, j) for i in clique_groups, j in clique_groups]
            for ii in 1:size(clique_groups,1)
                  group_mat  = pair_matrix(clique_groups[ii])
                  clique_mat = Gamma_hat[group_mat]
                  @constraint(model, Symmetric(clique_mat) in PSDCone())
            end

            #=
            # Now, we have three steps:
            # 1. create a PSD matrix variable for each group
            clique_mats = Dict{Integer, Matrix{JuMP.VariableRef}}(1 => @variable(model, [1:length(clique_groups[1]),1:length(clique_groups[1])], PSD))
            for ii in 2:size(clique_groups,1)
                  numbus_group    = length(clique_groups[ii])
                  clique_mats[ii] = @variable(model, [1:numbus_group,1:numbus_group], PSD)
            end

            # 2. Link the groups!
            tree = PowerModels._prim(PowerModels._overlap_graph(clique_groups))
            overlapping_pairs = [Tuple(CartesianIndices(tree)[i]) for i in (LinearIndices(tree))[findall(x->x!=0, tree)]]
            pair_matrix(clique_groups) = [(i, j) for i in clique_groups, j in clique_groups]
            for (i, j) in overlapping_pairs
                  gi,    gj    = clique_groups[i], clique_groups[j]
                  var_i, var_j = clique_mats[i],   clique_mats[j]
                  Gi,    Gj    = pair_matrix(gi),  pair_matrix(gj)
                  overlap_i, overlap_j = PowerModels._overlap_indices(Gi, Gj)
                  indices = zip(overlap_i, overlap_j)
                  for (idx_i, idx_j) in indices
                        JuMP.@constraint(model, var_i[idx_i] == var_j[idx_j])
                  end
            end

            # 3. Place into the larger matrix [Matrix{JuMP.VariableRef}(undef,...)] -- > does not work
            Gamma_hat = Matrix{JuMP.AffExpr}(undef, (num_ReLU + num_ReLU + 2*num_buses), (num_ReLU + num_ReLU + 2*num_buses))
            for ii in 1:length(clique_groups)
                  clique_mat_inds = pair_matrix(clique_groups[ii])
                  for jj = 1:size(clique_mat_inds,1)
                        for kk = 1:size(clique_mat_inds,2)
                              Gamma_hat[clique_mat_inds[jj,kk][1],clique_mat_inds[jj,kk][2]] = clique_mats[ii][jj,kk]
                        end
                  end
            end
            @constraint(model, Gamma_hat[1,1]  == 1)

            # now, set all "#undef" values to 0
            for ii in 1:(num_ReLU + num_ReLU + 2*num_buses)
                  for jj in 1:(num_ReLU + num_ReLU + 2*num_buses)
                        if !isassigned(Gamma_hat,ii,jj)
                              Gamma_hat[ii,jj] = 0
                        end
                  end
            end
            =#
      else
            # Define the *full* SDP Matrices
            @variable(model, Gamma_hat[1:(num_ReLU + num_ReLU + 2*num_buses), 1:(2*num_buses + num_ReLU + num_ReLU)], PSD)
            @constraint(model, Gamma_hat[1,1]  == 1)
      end

      # Define sub-variables
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

      # Add linear constraints
      @constraint(model, A*u .<= b)

      #@constraint(model, 0 .<= eta)
      #@constraint(model, 0 .<= delta)
      #@constraint(model, 0 .<= B)

      # Define an ouput vector and parse
      @expression(model, y, W1_bar*x + J0*v + b1_bar) # y = W1_bar*x + b1_bar + J0*v
      Vm2_nn  = y[                             (1:num_buses)]
      P_nn    = y[num_buses                 .+ (1:num_buses)]
      Q_nn    = y[2*num_buses               .+ (1:num_buses)]
      Ift2_nn = y[3*num_buses               .+ (1:num_lines)]
      Itf2_nn = y[3*num_buses + 1*num_lines .+ (1:num_lines)]

      # We need to build up the "Gamma = (u)(u^T)=uut" matrix
      # Gamma = [V    vxt   vbt;
      #          vxt' xxt   xbt;
      #          vbt' xbt'   B]
      # Gamma = [V          epsilon   gamma;
      #          epsilon'   eta       delta;
      #          gamma'     delta'    B]
      Gamma = Gamma_hat[2:end,2:end]

      # Loop to define the Omega matrix, but skip constraints we will not use
            # Alternative:  ->   Omega = A*Gamma*(A') - A*u*(b') - b*(A*u)' + b*b'
      @expression(model, GAp, Gamma*(A'))
      @expression(model, Au, A*u)

      for ii = 1:size(A,1)
            for jj = 1:size(A,1)
                  if ii <= jj
                        if (ii,jj) in initial_constraints
                              Omega_ij = AffExpr(0.0)
                              add_to_expression!(Omega_ij,b[ii]*b[jj])
                              add_to_expression!(Omega_ij,-b[ii]*Au[jj])
                              add_to_expression!(Omega_ij,-Au[ii]*b[jj])
                              add_to_expression!(Omega_ij,A[ii,:]'*GAp[:,jj])
                              @constraint(model, 0 <= verification_params[:SDP_constraint_tol] + Omega_ij)
                        end
                  end
            end
      end

      # Add key equality constraints (binary and quadratic ReLU reformulation)
      @constraint(model, diag(B) .== beta)
      for ii = 1:nx
            @constraint(model,delta[ii,ii] == x[ii])
            @constraint(model,  eta[ii,ii] ==  W0_bar[ii,:]'*epsilon[:,ii] + x[ii]*b0_bar[ii])
      end

      ###########################################################
      # Build Verification Objective Functions ##################
      ###########################################################
      Verification_Results = Dict()
      #
      # In this case, find the maximum error for a single line/bus
      if verification_index > 0
            # Test the type of verification
            if verification_routine == "V2"
                  # Squared_Voltage_Magnitude
                  bus      = verification_index
                  Mat      = ac_power_flow_model.bus_Vmag2_QMat[bus][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                  obj_up   = Vm2_nn[bus] - tr(Mat*V) # transformed from v'*Mat_ft*v
                  obj_down = -obj_up

            ###########################################
            elseif verification_routine == "P"
                  # P_Injection
                  bus      = verification_index
                  Mat      = ac_power_flow_model.bus_Pinj_QMat[bus][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                  obj_up   = P_nn[bus] - tr(Mat*V)
                  obj_down = -obj_up

            ###########################################
            elseif verification_routine == "Q"
                  # Q_Injection
                  bus      = verification_index
                  Mat      = ac_power_flow_model.bus_Qinj_QMat[bus][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                  obj_up   = Q_nn[bus] - tr(Mat*V)
                  obj_down = -obj_up

            ###########################################
            elseif verification_routine == "Ift2"
                  # Squared_Current_ft
                  line     = verification_index
                  Mat_ft   = ac_power_flow_model.line_Ift2_QMat[line][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                  obj_up   = Ift2_nn[line] - tr(Mat_ft*V)
                  obj_down = -obj_up

            elseif verification_routine == "Itf2"
                  # Squared_Current_tf
                  line     = verification_index
                  Mat_tf   = ac_power_flow_model.line_Itf2_QMat[line][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                  obj_up   = Itf2_nn[line] - tr(Mat_tf*V)
                  obj_down = -obj_up

            else
                  println("Verification routine not recognized!")
            end

            #Mat1      = ac_power_flow_model.bus_Pinj_QMat[1][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            #Mat2      = ac_power_flow_model.bus_Pinj_QMat[2][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            #Mat3      = ac_power_flow_model.bus_Pinj_QMat[3][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            #Mat4      = ac_power_flow_model.bus_Pinj_QMat[4][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            #Mat10     = ac_power_flow_model.bus_Pinj_QMat[10][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            #obj_up   = P_nn[1] + P_nn[2] + P_nn[3] + P_nn[4] + P_nn[10] - tr(Mat1*V) - tr(Mat2*V) - tr(Mat3*V) - tr(Mat4*V) - tr(Mat10*V)
            #obj_down = -obj_up

            # Solve down
            if error_type == "overestimation"
                  @objective(model, Max, obj_up)
            elseif error_type == "underestimation"
                  @objective(model, Max, obj_down)
            else
                  # incorrect objective
            end
            
            optimize!(model)
            println("=============================================================")
            println(termination_status(model),". ",primal_status(model),". objective value: ", objective_value(model))
            println("=============================================================")

            # parse
            if sparse_SDP == true
                  # Define outputs
                  Results  = Dict(
                        :Gamma_hat => value.(Gamma_hat),
                        :Gamma => [value.(V)          value.(epsilon)   value.(gamma);
                                    value.(epsilon)'   value.(eta)       value.(delta);
                                    value.(gamma)'     value.(delta)'    value.(B)],
                        :clique_groups => clique_groups,
                        :adj           => adj, 
                        :chordal_adj   => chordal_adj,
                        :u       => value.(u),
                        :A       => A,
                        :b       => b,
                        :knowninds => knowninds,
                        :GH      => Gamma_hat)
                        #:clique_mats => clique_mats)

            else
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
                        :u       => value.(u),
                        :A       => A,
                        :b       => b,
                        :Gamma_hat => value.(Gamma_hat),
                        :Gamma     => [value.(V)          value.(epsilon)   value.(gamma);
                                       value.(epsilon)'   value.(eta)       value.(delta);
                                       value.(gamma)'     value.(delta)'    value.(B)],
                        :sorted_Omega => 0)
                  
                  # prepare a model to return
                  verification_model = Dict(
                        :model     => model,
                        :A         => A,
                        :b         => b,
                        :Gamma_hat => Gamma_hat,
                        :GAp       => GAp,
                        :Au        => Au,
                        :B         => B,
                        :V         => V,
                        :x         => x,
                        :v         => v,
                        :beta      => beta,
                        :eta       => eta,
                        :epsilon   => epsilon,
                        :gamma     => gamma,
                        :delta     => delta,
                        :u         => u,
                        :W0_bar    => W0_bar,
                        :b0_bar    => b0_bar)

                  # print and update output
                  verification_params, Results = parse_and_print(Results, verification_model, verification_params)

                  # constrain the voltage diagonals
                  skip_for_plotting = 0
                  try
                        skip_for_plotting = verification_params[:skip_for_plotting] 
                  catch
                        skip_for_plotting = 0
                  end
                  
                  # we might want to skip if we want to show the power of adding a few constraints at a time
                  if skip_for_plotting == 1
                        # skip
                        verification_params[:voltage_constraints] = initial_constraints[1]
                        verification_params[:eta_constraints]     = initial_constraints[1]
                  else
                        for ii = 1:nv
                              if ii == 1
                                    verification_params[:voltage_constraints] = [(ii,ii + nv)]
                              else
                                    verification_params[:voltage_constraints] = [verification_params[:voltage_constraints]; [(ii,ii + nv)]]
                              end

                              # unclude diagonals? -- not very helpful..
                                    # verification_params[:voltage_constraints] = [verification_params[:voltage_constraints]; [(ii,ii)]]
                                    # verification_params[:voltage_constraints] = [verification_params[:voltage_constraints]; [(ii+length(V_max_full),ii+length(V_max_full))]]
                        end

                        # constrain the eta matrix diagonals
                        i_start_ind = size(Ab,1) + nx
                        j_start_ind = size(Ab,1) + nx + nb + nx
                        for ii = 1:nx
                              if ii == 1
                                    verification_params[:eta_constraints] = [(i_start_ind + ii,j_start_ind + ii)]
                              else
                                    verification_params[:eta_constraints] = [verification_params[:eta_constraints]; [(i_start_ind + ii,j_start_ind + ii)]]
                              end
                        end
                  end
            end
      else
            @warn("Verification index not (yet) recognized!")
      end

      # output
      return Results, verification_model, verification_params
end

function iterate_strong_SDP_verification(verification_model::Dict, verification_params::Dict)
      # parse the model
      model     = verification_model[:model]
      A         = verification_model[:A]
      b         = verification_model[:b]
      Gamma_hat = verification_model[:Gamma_hat]
      GAp       = verification_model[:GAp]
      Au        = verification_model[:Au]
      B         = verification_model[:B]
      V         = verification_model[:V]
      x         = verification_model[:x]
      v         = verification_model[:v]
      beta      = verification_model[:beta]
      eta       = verification_model[:eta]
      epsilon   = verification_model[:epsilon]
      gamma     = verification_model[:gamma]
      delta     = verification_model[:delta]
      u         = verification_model[:u]
      W0_bar    = verification_model[:W0_bar]
      b0_bar    = verification_model[:b0_bar]

      # constraints
      nnc   = verification_params[:num_new_constraints]
      pnc   = verification_params[:potential_new_constraints]
      vcs   = verification_params[:voltage_constraints]
      ecs   = verification_params[:eta_constraints]
      itcnt = verification_params[:iteration]

      # Set tolerances
      mosek_tols = verification_params[:mosek_tols]
      set_optimizer_attribute(model, "QUIET", false)
      set_optimizer_attribute(model, "MSK_IPAR_LOG_CUT_SECOND_OPT",       0)
      set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS",   mosek_tols)
      set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS",   mosek_tols)
      set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", mosek_tols)
      set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_MU_RED",  mosek_tols)
      set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_INFEAS",  mosek_tols)

      # Primary question: do we add new constraints? which ones?
      for ii = 1:size(A,1)
            for jj = 1:size(A,1)
                  if ii <= jj
                        if ((ii,jj) in pnc[1:nnc]) || (((ii,jj) in vcs) && (itcnt == 2)) || (((ii,jj) in ecs) && (itcnt == 2)) 
                              Omega_ij = AffExpr(0.0)
                              add_to_expression!(Omega_ij,b[ii]*b[jj])
                              add_to_expression!(Omega_ij,-b[ii]*Au[jj])
                              add_to_expression!(Omega_ij,-Au[ii]*b[jj])
                              add_to_expression!(Omega_ij,A[ii,:]'*GAp[:,jj])
                              @constraint(model, 0 <= verification_params[:SDP_constraint_tol] + Omega_ij)
                        end
                  end
            end
      end

      # everything after this point is the same
      optimize!(model)
      println("=============================================================")
      println(termination_status(model),". ",primal_status(model),". objective value: ", objective_value(model))
      println("=============================================================")
      # a nice addition: solution_summary(model)

      # define outputs
      Results = Dict(
            :obj       => objective_value(model),
            :B         => value.(B),
            :V         => value.(V),
            :x         => value.(x),
            :v         => value.(v),
            :beta      => value.(beta),
            :eta       => value.(eta),
            :epsilon   => value.(epsilon),
            :gamma     => value.(gamma),
            :delta     => value.(delta),
            :u         => value.(u),
            :Gamma_hat => value.(Gamma_hat),
            :Gamma     => [value.(V)         value.(epsilon)   value.(gamma);
                           value.(epsilon)'   value.(eta)       value.(delta);
                           value.(gamma)'     value.(delta)'    value.(B)],
            :sorted_Omega => 0)
      
      verification_model = Dict(
            :model     => model,
            :A         => A,
            :b         => b,
            :Gamma_hat => Gamma_hat,
            :GAp       => GAp,
            :Au        => Au,
            :B         => B,
            :V         => V,
            :x         => x,
            :v         => v,
            :beta      => beta,
            :eta       => eta,
            :epsilon   => epsilon,
            :gamma     => gamma,
            :delta     => delta,
            :u         => u,
            :W0_bar    => W0_bar,
            :b0_bar    => b0_bar)

      # print and log some results
      verification_params, Results = parse_and_print(Results, verification_model, verification_params)

      return Results, verification_model, verification_params
end

function stt_algorithm(num_runs::Int64, ac_power_flow_model::ac_power_flow_model_instance, ac_nn_model::ac_nn_model_instance, verification_params::Dict)
      # solve the initial SDP
      Results, verification_model, verification_params = strong_SDP_verification(ac_power_flow_model, ac_nn_model, verification_params);

      # store the nncs
      nncs_temp = copy(verification_params[:num_new_constraints])

      # jump into the loop
      for ii = 2:num_runs
            if ii == 2
                  verification_params[:num_new_constraints] = 1
            else
                  verification_params[:num_new_constraints] = nncs_temp
            end

            # verify
            Results, verification_model, verification_params = iterate_strong_SDP_verification(verification_model, verification_params);
      end

      # parse
      sttlog = filterlog_stt(verification_params)

      # output
      return sttlog
end

function stt_algorithm_plot(num_runs::Int64, ac_power_flow_model::ac_power_flow_model_instance, ac_nn_model::ac_nn_model_instance, verification_params::Dict)
      # solve the initial SDP
      Results, verification_model, verification_params = strong_SDP_verification(ac_power_flow_model, ac_nn_model, verification_params);
      gr()

      # colors
      c1 = 165/256
      c2 = 42/256
      c3 = 42/256
      redd = RGB(c1,c2,c3)

      display(plot([verification_params[:solutions][1][:total_run_time]],[Results[:obj]],seriestype = :scatter,ylabel = "Error Bound (ẽ)", color = :black,linewidth=1.5, xlim = [0.08; 50], ylim = [10^-1.6; 10^2.2], xticks = [10^(-2); 10^(-1); 10^0; 10^1], yaxis = :log, xaxis = :log10,label = "",xlabel = "time (sec)"))
      t1 = verification_params[:solutions][1][:total_run_time]
      o1 = Results[:obj]
      t2 = t1
      o2 = o1

      # store the nncs
      nncs_temp = copy(verification_params[:num_new_constraints])

      # jump into the loop
      for ii = 2:num_runs
            if ii == 2
                  verification_params[:num_new_constraints] = 1
            else
                  verification_params[:num_new_constraints] = nncs_temp
            end
            t1 = copy(t2)
            o1 = copy(o2)

            # verify
            Results, verification_model, verification_params = iterate_strong_SDP_verification(verification_model, verification_params);
            display(plot!([verification_params[:solutions][ii][:total_run_time]],[Results[:obj]],seriestype = :scatter,ylabel = "Error Bound (ẽ)", color = :black,linewidth=1.5, yaxis = :log, xaxis = :log10,label = "",xlabel = "time (sec)"))
            t2 = verification_params[:solutions][ii][:total_run_time]
            o2 = Results[:obj]
            if ii == 2
                  display(plot!([t1;t2],[o1;o2],color = 1,linewidth=1.5,label = "STT iterations (MOSEK)"))
            else
                  display(plot!([t1;t2],[o1;o2],color = 1,linewidth=1.5,label = ""))
            end
            #p1 = plot!(results_test1[case][:sttlog_timelog], results_test1[case][:sttlog_boundlog], ylabel = "Error Bound (ẽ)", xlabel = "", color = 1,linewidth=1.5, label = "STT iteration", xlim = [0.007; 15], xticks = [10^(-2); 10^(-1); 10^0; 10^1])
            #p1 = plot!(results_test1[case][:sttlog_timelog], results_test1[case][:sttlog_boundlog], seriestype = :scatter, ylabel = "Error Bound (ẽ)", xlabel = "", color = :black, markersize=1.75, label = "")  

      end

      # parse
      sttlog = filterlog_stt(verification_params)

      # output
      return sttlog
end

# determine the minimum number of constraints necessary for strong duality to hold
function necessary_strong_constraints(ac_power_flow_model::ac_power_flow_model_instance, verification_params::Dict)

      # parse 
      verification_routine = verification_params[:verification_routine]
      verification_index   = verification_params[:verification_index]

      if verification_routine == "V2"
            # Squared_Voltage_Magnitude
            bus          = verification_index
            Mat          = ac_power_flow_model.bus_Vmag2_QMat[bus][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            mat_nnz_inds = Tuple.(findall(x->x!=0, Mat))
      ###########################################
      elseif verification_routine == "P"
            # P_Injection
            bus          = verification_index
            Mat          = ac_power_flow_model.bus_Pinj_QMat[bus][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            mat_nnz_inds = Tuple.(findall(x->x!=0, Mat))
      ###########################################
      elseif verification_routine == "Q"
            # Q_Injection
            bus          = verification_index
            Mat          = ac_power_flow_model.bus_Qinj_QMat[bus][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            mat_nnz_inds = Tuple.(findall(x->x!=0, Mat))
      ###########################################
      elseif verification_routine == "Ift2"
            # Squared_Current_ft
            line         = verification_index
            Mat_ft       = ac_power_flow_model.line_Ift2_QMat[line][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            mat_nnz_inds = Tuple.(findall(x->x!=0, Mat_ft))
      elseif verification_routine == "Itf2"
            # Squared_Current_tf
            line         = verification_index
            Mat_tf       = ac_power_flow_model.line_Itf2_QMat[line][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            mat_nnz_inds = Tuple.(findall(x->x!=0, Mat_tf))
      else
            println("Verification routine not recognized!")
      end

      # remove the constraints associated with "lower diagonals
      ninds = length(mat_nnz_inds)
      for ii in 1:ninds
            jj = ninds - ii + 1
            tup = mat_nnz_inds[jj]
            if tup[2] < tup[1]
                  deleteat!(mat_nnz_inds,jj)
            end
      end

      # the vector "mat_nnz_inds" will constrain upper limits only -- add lower limits
      scalar_addition  = 2*ac_power_flow_model.num_buses-1
      mat_nnz_inds_aug = deepcopy(mat_nnz_inds)
      ninds            = length(mat_nnz_inds_aug)
      for ii in 1:ninds
            append!(mat_nnz_inds_aug,[(mat_nnz_inds[ii][1] + scalar_addition,mat_nnz_inds[ii][2] + scalar_addition)])
            append!(mat_nnz_inds_aug,[(mat_nnz_inds[ii][1],mat_nnz_inds[ii][2] + scalar_addition)])
      end

      # upate the parameters
      verification_params[:initial_constraints] = mat_nnz_inds_aug
      verification_params[:active_constraints]  = mat_nnz_inds_aug

      return verification_params
end

function parse_and_print(Results::Dict, verification_model::Dict, verification_params::Dict)
      # inequality violations
      Gamma = Results[:Gamma];
      A     = verification_model[:A];
      b     = verification_model[:b];
      u     = Results[:u];
      Omega = A*Gamma*(A') - A*u*(b') - b*(A*u)' + b*b';

      # update the constraint set 
      if verification_params[:iteration] == 1
            verification_params[:active_constraints] = copy(verification_params[:initial_constraints])
      elseif verification_params[:iteration] == 2
            # we add the other votlage constraints here
            verification_params[:active_constraints] = [verification_params[:active_constraints] ;verification_params[:potential_new_constraints][1:verification_params[:num_new_constraints]]]
            vcs                                      = verification_params[:voltage_constraints]
            ecs                                      = verification_params[:eta_constraints]
            verification_params[:active_constraints] = [verification_params[:active_constraints]; vcs; ecs]
      else
            verification_params[:active_constraints] = [verification_params[:active_constraints] ;verification_params[:potential_new_constraints][1:verification_params[:num_new_constraints]]]
      end

      # normalize?
      normalize_Omega = false # hardcode
      if normalize_Omega == true
            Omega_est = abs.(A*(u*u')*(A') - A*u*(b') - b*(A*u)' + b*b');
            Omega_est[Omega_est .< 1e-5] .= 1.0
            Omega_normal = Omega./Omega_est;
      else
            Omega_normal = Omega
      end

      # sort, print, and plot
      sorted_Omega = sort(vec(Omega_normal))
      largest_sorted_Omega = round.(sorted_Omega[1:2:5], sigdigits = 5)
      println("largest Ω violations: $largest_sorted_Omega")
      num_plot_values = Int(min(1e4,round(length(sorted_Omega)/2)))
      if verification_params[:iteration] == 1
            #display(plot(sorted_Omega[1:2:num_plot_values]))
      else
            #display(plot!(sorted_Omega[1:2:num_plot_values]))
      end
      
      # this is super overkill, but "Omega_UD" is the Omega matrix with
      # all zeros below the diagonal -- we pull the indices from here
      num_potential_new_constraints = 1000
      nOmega                    = size(Omega_normal,1)
      Omega_UD                  = Omega_normal - [ x>y ? Omega_normal[x,y] : 0 for x in 1:nOmega, y in 1:nOmega]
      potential_new_constraints = Tuple.(argsmallest(Omega_UD,num_potential_new_constraints))

      # Build the matrix "Gamma_xxhat", which is a key subset of Gamma_hat
      Gamma_xxhat = [1            Results[:v]'         Results[:x]';
                    Results[:v]   Results[:V]          Results[:epsilon];
                    Results[:x]   Results[:epsilon]'   Results[:eta]]

      # compute eigenvalues of voltage
      run_extra_eigs = 1
      if run_extra_eigs == 1
            eigs_voltage  = abs.(eigvals(Results[:V]))
            eig12_voltage = round(eigs_voltage[end]/eigs_voltage[end-1], sigdigits = 5)
            eig13_voltage = round(eigs_voltage[end]/eigs_voltage[end-2], sigdigits = 5)
            eigs_eta  = abs.(eigvals(Results[:eta]))
            eig12_eta = round(eigs_eta[end]/eigs_eta[end-1], sigdigits = 5)
            eig13_eta = round(eigs_eta[end]/eigs_eta[end-2], sigdigits = 5)
      end

      # compute eigenvalues of binary
      eigs_binary  = abs.(eigvals(Results[:B]))
      eig12_binary = round(eigs_binary[end]/eigs_binary[end-1], sigdigits = 5)
      eig13_binary = round(eigs_binary[end]/eigs_binary[end-2], sigdigits = 5)

      # compute eigenvalues of the full matrix: Gamma_xxhat (see paper)
      eigs_full     = eigvals(Results[:Gamma_hat]) # Gamma_xxhat
      eigs_full_abs = abs.(eigs_full)
      eig12_full    = round(eigs_full_abs[end]/eigs_full_abs[end-1], sigdigits = 5)
      eig13_full    = round(eigs_full_abs[end]/eigs_full_abs[end-2], sigdigits = 5)

      # print the results
      println("============ Eigenvalue Summary ==============================")
      if run_extra_eigs == 1
            println("voltage eigenvalue ratio: (λ1/λ2) $eig12_voltage and (λ1/λ3) $eig13_voltage")
            println("eta eigenvalue ratio: (λ1/λ2) $eig12_eta and (λ1/λ3) $eig13_eta")
      end
      println("binary eigenvalue ratio: (λ1/λ2) $eig12_binary and (λ1/λ3) $eig13_binary")
      println("full eigenvalue ratio: (λ1/λ2) $eig12_full and (λ1/λ3) $eig13_full")

      println("============ Feasibility summary ==============================")
      smallest_eig         = round(minimum(eigs_full), sigdigits = 5)
      smallest_linear_ineq = round(minimum(b - A*value.(u)), sigdigits = 5)
      smallest_matrix_ineq = round(minimum(Omega[CartesianIndex.(verification_params[:active_constraints])] .+ verification_params[:SDP_constraint_tol]), sigdigits = 5)

      nx                         = size(Results[:B],1)
      largest_eq_violation       = zeros(3*nx,1)
      largest_eq_violation[1:nx] = abs.(diag(Results[:B]) - Results[:beta])
      ij = nx+1
      for ii = 1:nx
            largest_eq_violation[ij] = abs(Results[:delta][ii,ii] - Results[:x][ii])
            ij = ij + 1
            largest_eq_violation[ij] = abs(Results[:eta][ii,ii] -  (verification_model[:W0_bar][ii,:]'*Results[:epsilon][:,ii] + Results[:x][ii]*verification_model[:b0_bar][ii]))
            ij = ij + 1
      end
      largest_eq_violation = round(maximum(abs.(largest_eq_violation)), sigdigits = 5)
      gap                  = JuMP.relative_gap(verification_model[:model])

      @info("Smallest λ: $smallest_eig. Linear ineq: $smallest_linear_ineq. Matrix ineq: $smallest_matrix_ineq. Eq: $largest_eq_violation Gap: $gap.")

      # tests -- not really needed..
      t1 = smallest_eig         > -1e-10
      t2 = smallest_linear_ineq > -1e-10
      t3 = smallest_matrix_ineq > -1e-10
      t4 = largest_eq_violation > 1e-10
      t5 = gap                  > 1e-3

      # if t1 && t2 && t3 && t4 && t5
      #      @info("All good")
      # end
      
      # update potential new constraints
      verification_params[:potential_new_constraints] = potential_new_constraints

      # Query user
      @info("Run again with additonal constraints? How many? (Up to $num_potential_new_constraints)")

      # define the total runtime
      total_run_time = solve_time(verification_model[:model])
      for ii = 1:(verification_params[:iteration]-1)
            total_run_time = total_run_time + verification_params[:solutions][ii][:run_time]
      end

      verification_params[:solutions][verification_params[:iteration]] = Dict(
            :total_run_time  => total_run_time,
            :run_time        => solve_time(verification_model[:model]),
            :objective_value => objective_value(verification_model[:model]),
            :eig_ratio_12    => eig12_full,
            :eig_ratio_13    => eig13_full,
            )
      
      # update the iteration count
      verification_params[:iteration] = verification_params[:iteration] + 1
      Results[:sorted_Omega]          = sorted_Omega

      return verification_params, Results
end

function miqp_verification(ac_power_flow_model::ac_power_flow_model_instance, ac_nn_model::ac_nn_model_instance, verification_params::Dict, timelim::Float64)
      # Build model
      model = Model(Gurobi.Optimizer)
      empty!(model)

      # Model Settings
      set_optimizer_attribute(model, "NonConvex", 2)
      set_optimizer_attribute(model, "MIPGap", 0.001)
      set_optimizer_attribute(model, "TimeLimit", timelim)
            # print: set_optimizer_attribute(model, "ResultFile", "MySolution.sol")

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
      M_max    = ac_nn_model.M_max
      M_min    = ac_nn_model.M_min
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
      Ift2_min = -5e-2*ones(length(Ift2_max),1) # previously: 0*Ift2_max --- -1e-2 for numerics
      Itf2_min = -5e-2*ones(length(Ift2_max),1) # previously: 0*Itf2_max --- -1e-2 for numerics
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

      # sanitize
      num_params_below_threshold = sum(0 .< abs.(b) .< setzero_threshold) + sum(0 .< abs.(A) .< setzero_threshold)
      @info("Number of model parameters below $setzero_threshold: $num_params_below_threshold")

      @variable(model, v[1:(2*num_buses-1)])
      @variable(model, x[1:num_ReLU])
      @variable(model, beta[1:num_ReLU], binary = true)

      # Add linear constraints
      u = [v; x; beta]
      @constraint(model, A*u .<= b)

      # Define an ouput vector and parse
      @expression(model, y, W1_bar*x + J0*v + b1_bar) # y = W1_bar*x + b1_bar + J0*v
      Vm2_nn  = y[                             (1:num_buses)]
      P_nn    = y[num_buses                 .+ (1:num_buses)]
      Q_nn    = y[2*num_buses               .+ (1:num_buses)]
      Ift2_nn = y[3*num_buses               .+ (1:num_lines)]
      Itf2_nn = y[3*num_buses + 1*num_lines .+ (1:num_lines)]

      ###########################################################
      # Build Verification Objective Functions ##################
      ###########################################################
      Verification_Results = Dict()
      #
      # In this case, find the maximum error for a single line/bus
      if verification_index > 0
            # Test the type of verification
            if verification_routine == "V2"
                  # Squared_Voltage_Magnitude
                  bus      = verification_index
                  Mat      = ac_power_flow_model.bus_Vmag2_QMat[bus][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                  obj_up   = Vm2_nn[bus] - (v'Mat*v) # transformed from v'*Mat_ft*v
                  obj_down = -obj_up

            ###########################################
            elseif verification_routine == "P"
                  # P_Injection
                  bus      = verification_index
                  Mat      = ac_power_flow_model.bus_Pinj_QMat[bus][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                  obj_up   = P_nn[bus] - (v'Mat*v)
                  obj_down = -obj_up

            ###########################################
            elseif verification_routine == "Q"
                  # Q_Injection
                  bus      = verification_index
                  Mat      = ac_power_flow_model.bus_Qinj_QMat[bus][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                  obj_up   = Q_nn[bus] - (v'Mat*v)
                  obj_down = -obj_up

            ###########################################
            elseif verification_routine == "Ift2"
                  # Squared_Current_ft
                  line     = verification_index
                  Mat_ft   = ac_power_flow_model.line_Ift2_QMat[line][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                  obj_up   = Ift2_nn[line] - (v'Mat_ft*v)
                  obj_down = -obj_up

            elseif verification_routine == "Itf2"
                  # Squared_Current_tf
                  line     = verification_index
                  Mat_tf   = ac_power_flow_model.line_Itf2_QMat[line][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
                  obj_up   = Itf2_nn[line] - (v'Mat_tf*v)
                  obj_down = -obj_up
            else
                  println("Verification routine not recognized!")
            end

            #Mat1      = ac_power_flow_model.bus_Pinj_QMat[1][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            #Mat2      = ac_power_flow_model.bus_Pinj_QMat[2][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            #Mat3      = ac_power_flow_model.bus_Pinj_QMat[3][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            #Mat4      = ac_power_flow_model.bus_Pinj_QMat[4][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            #Mat10     = ac_power_flow_model.bus_Pinj_QMat[10][Not(num_buses+ref_bus),Not(num_buses+ref_bus)]
            #obj_up   = P_nn[1] + P_nn[2] + P_nn[3] + P_nn[4] + P_nn[10] - (v'Mat1*v) - (v'Mat2*v) - (v'Mat3*v) - (v'Mat4*v) - (v'Mat10*v)
            #obj_down = -obj_up

            # Solve down
            if error_type == "overestimation"
                  @objective(model, Max, obj_up)

            elseif error_type == "underestimation"
                  @objective(model, Max, obj_down)

            else
                  # incorrect objective
            end

            # log structures
            global timelog  = []
            global bestlog  = []
            global boundlog = []

            # plot...?
            try 
                  if verification_params[:plot_miqp] == true
                        MOI.set(model, Gurobi.CallbackFunction(), callback_log_plot)
                  else
                        MOI.set(model, Gurobi.CallbackFunction(), callback_log)
                  end
            catch
                  MOI.set(model, Gurobi.CallbackFunction(), callback_log)
            end 

            # optimize
            optimize!(model)

            # filter and record
            miqplog = filterlog_miqp(timelog,bestlog,boundlog)

            # optimize!(model)
            println("=============================================================")
            println(termination_status(model),". ",primal_status(model),". objective value: ", objective_value(model))
            println("=============================================================")
      end

      return miqplog
end

function argsmallest(A::AbstractArray{T,N}, n::Integer) where {T,N}
      # should someone ask more elements than array size, just sort array
      if n>= length(vec(A))
            ind=collect(1:length(vec(A)))
            ind=sortperm(A[ind])
            return CartesianIndices(A)[ind]
      end

      # otherwise 
      ind=collect(1:n)
      mymax=maximum(A[ind])
      for j=n+1:length(vec(A))
            if A[j]<mymax
                  getout=findmax(A[ind])[2]
                  ind[getout]=j
                  mymax=maximum(A[ind])
            end
      end
      ind=ind[sortperm(A[ind])]
      
      return CartesianIndices(A)[ind]
  end

# build graph

function buildgraphadj(numedges::Int,knowninds::Matrix)
      adj = spzeros(numedges,numedges)
      for ii in 1:size(knowninds,1)
            if knowninds[ii,2] < knowninds[ii,1]
                  adj[knowninds[ii,2],knowninds[ii,1]] = 1.0
            elseif knowninds[ii,2] > knowninds[ii,1]
                  adj[knowninds[ii,1],knowninds[ii,2]] = 1.0
            else
                  # diagonals -- skip for the adj matrix
            end
      end

      # now, loop over the matrix itself
      for ii in 1:(size(adj,1)-1) # skip the last
            if (sum(adj[ii,:]) > 0) || (sum(adj[:,ii]) > 0) 
                  # this bus is already connected to something
            else
                  adj[ii,ii+1] = 1.0
                  println("edge added!!!")
            end
      end

      # update the L-Diag
      adj = adj + adj'

      # perform a chordal extension
      chordal_adj, peo = chordal_extension(adj)
      #clique_groups    = PowerModels._maximal_cliques(chordal_adj,_mcs(chordal_adj))
      clique_groups    = PowerModels._maximal_cliques(chordal_adj, peo)

      # output
      return clique_groups, adj, chordal_adj

end

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
      cadj_preperm = sparse([f_idx;t_idx], [t_idx;f_idx], ones(2*length(f_idx)), nb, nb)
      cadj         = cadj_preperm[q, q] # revert to original bus ordering (invert cholfact permutation)

      # cadj = copy(adj) # %%%%%%%%%%%%%%%%% ---

      return cadj, p, cadj_preperm
end
"""
    mc = _maximal_cliques(cadj, peo)
Given a chordal graph adjacency matrix and perfect elimination
ordering, return the set of maximal cliques.
"""
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

_maximal_cliques(cadj::SparseMatrixCSC) = _maximal_cliques(cadj, _mcs(cadj))

"""
    peo = _mcs(A)
Maximum cardinality search for graph adjacency matrix A.
Returns a perfect elimination ordering for chordal graphs.
"""
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

function minnz(A)
      Aabs = abs.(A)
      Aabs[Aabs .== 0] .= Inf

      (val,inds) = findmin(Aabs)

      return val, inds
end

using Base.Sort

function smallestn(A, n)
      Aabs = abs.(A)
      Aabs[Aabs .== 0] .= Inf

      return sort(Aabs[:]; alg=Sort.PartialQuickSort(n))[1:n]
end

function callback_log(cb_data, cb_where::Cint)
      MIP = 3
      if cb_where == MIP
            # println(cb_where)
            runtimeP = Ref{Cdouble}()
            objbstP  = Ref{Cdouble}()
            objbndP  = Ref{Cdouble}()
            GRBcbget(cb_data, cb_where, GRB_CB_RUNTIME, runtimeP)
            GRBcbget(cb_data, cb_where, GRB_CB_MIP_OBJBST, objbstP)
            GRBcbget(cb_data, cb_where, GRB_CB_MIP_OBJBND, objbndP)
            best  = objbstP[]
            bound = objbndP[]

            # push data
            push!(bestlog, best)
            push!(boundlog, bound)
            push!(timelog, runtimeP[])
            # don't save the gap:           
                  # gap = abs((objbstP[] - objbndP[]) / objbstP[])
                  # push!(gaplog, gap)
      end
      return
  end

  
function callback_log_plot(cb_data, cb_where::Cint)
      MIP = 3
      if cb_where == MIP
            # println(cb_where)
            runtimeP = Ref{Cdouble}()
            objbstP  = Ref{Cdouble}()
            objbndP  = Ref{Cdouble}()
            GRBcbget(cb_data, cb_where, GRB_CB_RUNTIME, runtimeP)
            GRBcbget(cb_data, cb_where, GRB_CB_MIP_OBJBST, objbstP)
            GRBcbget(cb_data, cb_where, GRB_CB_MIP_OBJBND, objbndP)
            best  = objbstP[]
            bound = objbndP[]

            # push data
            push!(bestlog, best)
            push!(boundlog, bound)
            push!(timelog, runtimeP[])

            # plot
            redd = RGB(165/256,42/256,42/256)
            if size(bestlog,1) <= 1
                  display(plot!(timelog, boundlog, label = "MIQP (Gurobi)", linewidth=4, opacity=.75, color = redd))
            else
                  display(plot!(timelog, boundlog, label = "",linewidth=4, opacity=.75, color = redd))
            end

            # don't save the gap:           
                  # gap = abs((objbstP[] - objbndP[]) / objbstP[])
                  # push!(gaplog, gap)
      end
      return
end

function filterlog_miqp(timelog,bestlog,boundlog)
      # filter out the "-1.0e100" values (no solution found)
      inds           = abs.(bestlog) .> 1e15
      bestlog[inds] .= 0
      max_ind        = argmax(abs.(bestlog))
      bestlog[inds] .= bestlog[max_ind]
      gaplog         = 100*abs.((bestlog - boundlog) ./ bestlog)

      miqplog = Dict(
            :gaplog   => convert(Array{Float64},gaplog),
            :timelog  => convert(Array{Float64},timelog),
            :bestlog  => convert(Array{Float64},bestlog),
            :boundlog => convert(Array{Float64},boundlog))

      return miqplog
end

function filterlog_stt(verification_params)
      itcnt    = verification_params[:iteration]-1
      timelog  = []
      boundlog = []
      eig12log = []
      eig13log = []
      for ii in 1:itcnt
            push!(timelog,verification_params[:solutions][ii][:total_run_time])
            push!(boundlog,verification_params[:solutions][ii][:objective_value])
            push!(eig12log,verification_params[:solutions][ii][:eig_ratio_12])
            push!(eig13log,verification_params[:solutions][ii][:eig_ratio_13])
      end

      sttlog = Dict(
            :timelog  => convert(Array{Float64},timelog),
            :boundlog => convert(Array{Float64},boundlog),
            :eig12log => convert(Array{Float64},eig12log),
            :eig13log => convert(Array{Float64},eig13log))

      return sttlog
end

function write_verificationdata(masterlog,testdata_file,cases)
      # Write to file
      fid = h5open(testdata_file, "w") do file
            for case in cases
                  casestr = string(case)
                  # stt
                  write(file, "sttlog_timelog_"*casestr,  masterlog[case][:sttlog][:timelog])
                  write(file, "sttlog_boundlog_"*casestr, masterlog[case][:sttlog][:boundlog])
                  write(file, "sttlog_eig12log_"*casestr, masterlog[case][:sttlog][:eig12log])
                  write(file, "sttlog_eig13log_"*casestr, masterlog[case][:sttlog][:eig13log])

                  # miqp
                  write(file, "miqplog_gaplog_"*casestr,  masterlog[case][:miqplog][:gaplog])
                  write(file, "miqplog_timelog_"*casestr, masterlog[case][:miqplog][:timelog])
                  write(file, "miqplog_bestlog_"*casestr, masterlog[case][:miqplog][:bestlog])
                  write(file, "miqplog_boundlog_"*casestr, masterlog[case][:miqplog][:boundlog])
            end
      end
end

function write_verificationdata_table(masterlog,testdata_file)
      # Write to file
      fid = h5open(testdata_file, "w") do file
            for ii = 1:5
                  iistr = string(ii)

                  # stt
                  write(file, "sttlog_timelog"*iistr,  masterlog[ii][:sttlog][:timelog])
                  write(file, "sttlog_boundlog"*iistr, masterlog[ii][:sttlog][:boundlog])
                  write(file, "sttlog_eig12log"*iistr, masterlog[ii][:sttlog][:eig12log])
                  write(file, "sttlog_eig13log"*iistr, masterlog[ii][:sttlog][:eig13log])

                  # miqp
                  write(file, "miqplog_gaplog"*iistr,   masterlog[ii][:miqplog][:gaplog])
                  write(file, "miqplog_timelog"*iistr,  masterlog[ii][:miqplog][:timelog])
                  write(file, "miqplog_bestlog"*iistr,  masterlog[ii][:miqplog][:bestlog])
                  write(file, "miqplog_boundlog"*iistr, masterlog[ii][:miqplog][:boundlog])
            end
      end
end