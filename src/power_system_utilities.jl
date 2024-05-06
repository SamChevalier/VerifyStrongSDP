# %% Build AC_PowerFlowModel and AC_NN_Model
function build_acpf_model(network_data::Dict,powerflow_network_file::String)

      # Network matrices
      Yb, NY, NYft, NYtf, E1, E2, E, YSftE1, YStfE2, NYImft, NYImtf, YLft, YLtf = network_structures(network_data)
  
      # Build Q-mats
      Mp, Mq, MV2, Mpft, Mptf, Mqft, Mqtf, MIft2, MItf2 = qcqp_matrices(Yb,E,network_data,YLft,YLtf,YSftE1,YStfE2)
  
      # Load power flow limit data
      c = h5open(powerflow_network_file, "r") do file
          global V_max      = read(file, "V_max")
          global V_min      = read(file, "V_min")
          global DTheta_max = read(file, "DTheta_max")
          global DTheta_min = read(file, "DTheta_min")
          global Pinj_max   = read(file, "Pinj_max")
          global Pinj_min   = read(file, "Pinj_min")
          global Qinj_max   = read(file, "Qinj_max")
          global Qinj_min   = read(file, "Qinj_min")
          global Ift2_max   = read(file, "Ift2_max")
          global Itf2_max   = read(file, "Itf2_max")
          global num_buses  = read(file, "num_buses")
          global num_lines  = read(file, "num_lines")
          global ref_bus    = read(file, "ref_bus")
      end
  
      # Build AC_PowerFlowModel
      ac_power_flow_model = ac_power_flow_model_instance(
          ref_bus,
          num_buses,
          num_lines,
          MV2,
          Mp,
          Mq,
          Mpft,
          Mqft,
          MIft2,
          Mptf,
          Mqtf,
          MItf2,
          V_max,
          V_min,
          Ift2_max,
          Itf2_max,
          Pinj_max,
          Pinj_min,
          Qinj_max,
          Qinj_min)
  
      # Output
      return ac_power_flow_model
  end

  # %% First, define necessary admittance structures
function network_structures(network_data::Dict)
      AdmitMat = PowerModels.calc_admittance_matrix(network_data)
      Yb       = AdmitMat.matrix
      Ybr      = real(Yb)
      Ybi      = imag(Yb)
      NY       = [Ybr -Ybi;
                 -Ybi -Ybr]

      # Now, compute line admittance matrix
      E, E1, E2, YL, YLft, YLtf, YSft, YStf, YlDC = network_incidence(network_data)

      # Update with Shunts
      YSftE1 = YSft*E1
      YStfE2 = YStf*E2

      # From -> To
      YLftr = real(YLft+YSftE1)
      YLfti = imag(YLft+YSftE1)
      NYft  = [YLftr -YLfti;
              -YLfti -YLftr]
      NYImft = [YLftr -YLfti;
                YLfti  YLftr]
      # To -> From
      YLtfr = real(YLtf+YStfE2)
      YLtfi = imag(YLtf+YStfE2)
      NYtf  = [YLtfr -YLtfi;
              -YLtfi -YLtfr]
      NYImtf = [YLtfr -YLtfi;
                YLtfi  YLtfr]

      return Yb, NY, NYft, NYtf, E1, E2, E, YSftE1, YStfE2, NYImft, NYImtf, YLft, YLtf, YlDC
end

# return the number of connections
function num_bus_connections(network_data)
      E, E1, E2, YL, YLft, YLtf, YSft, YStf, YlDC = network_incidence(network_data)
      
      return diag(Matrix(E'*E))
end

# build incidence matrix
function network_incidence(network_data::Dict)
      II     = Int64[]
      ii     = Int64[]
      JJ     = Int64[]
      V      = Int64[]
      BrList = Int64[]
      BrKeep = Int64[]
      YLi    = ComplexF64[]
      YSfti  = ComplexF64[]
      YStfi  = ComplexF64[]
      Ylx    = ComplexF64[]
        for (i,branch) in network_data["branch"]
           f_bus  = branch["f_bus"]
           t_bus  = branch["t_bus"]
           yff    = branch["g_fr"] + im*branch["b_fr"]
           ytt    = branch["g_to"] + im*branch["b_to"]
           ybr    = 1/(branch["br_r"] + im*branch["br_x"])
           ybr    = ybr/branch["tap"]
           brX    = branch["br_x"]  # DC power flow
           BrList = []

           if branch["br_status"] != 0
                push!(JJ, f_bus); push!(II, parse(Int,i)); push!(V,  1)
                push!(JJ, t_bus); push!(II, parse(Int,i)); push!(V,  -1)
                push!(ii,parse(Int,i))
                push!(YLi,ybr)
                # The following will combine parallel lines!!! -- Don't use
                  # push!(YLi,-Yb[f_bus,t_bus])
                push!(YSfti,yff)
                push!(YStfi,ytt)
                push!(Ylx,1/brX)

                # Update the branch list
                push!(BrList,[f_bus t_bus])
                push!(BrKeep,parse(Int,i))
            end
        end

      # Construct Incidence Matrix
      E    = sparse(II,JJ,V)
      YL   = sparse(ii,ii,YLi)
      YLE  = YL*E
      YLft = deepcopy(YLE)
      YLtf = deepcopy(-YLE)

      # What about transformer ratios?
      for (i,branch) in network_data["branch"]
            if branch["tap"] != 1
                  f_bus = branch["f_bus"]
                  t_bus = branch["t_bus"]

                  # The following is validated against Power Models. The value
                  # "YLft" already contains an inverted tap factor :)
                  YLft[parse(Int,i),f_bus]  = YLft[parse(Int,i),f_bus]*(1/branch["tap"])
                  YLft[parse(Int,i),t_bus]  = YLft[parse(Int,i),t_bus]
                  YLtf[parse(Int,i),f_bus]  = YLtf[parse(Int,i),f_bus]
                  YLtf[parse(Int,i),t_bus]  = YLtf[parse(Int,i),t_bus]*(branch["tap"])
            end
      end

      YSft = sparse(ii,ii,YSfti)
      YStf = sparse(ii,ii,YStfi)
      YlDC = sparse(ii,ii,Ylx)

      # Build Selection Matrices
      E2 = sparse(0.5*convert(Matrix{Float64},(abs.(E) - E)))
      E1 = sparse(0.5*convert(Matrix{Float64},(abs.(E) + E)))

      # now, we need to remove the appropriate rows and columns associated with
      # lines that are turned off (status = 0)
      BrList_sorted = sort(BrKeep)

      # Remove rows
      E    = E[BrList_sorted,:]
      E1   = E1[BrList_sorted,:]
      E2   = E2[BrList_sorted,:]
      YLft = YLft[BrList_sorted,:]
      YLtf = YLtf[BrList_sorted,:]
      YL   = YL[BrList_sorted,:]
      YSft = YSft[BrList_sorted,:]
      YStf = YStf[BrList_sorted,:]
      YlDC = YlDC[BrList_sorted,:]

      # Remove columns
      YL   = YL[:,BrList_sorted]
      YSft = YSft[:,BrList_sorted]
      YStf = YStf[:,BrList_sorted]
      YlDC = YlDC[:,BrList_sorted]

      return E, E1, E2, YL, YLft, YLtf, YSft, YStf, YlDC
end

# Build the matrices assocaited with the qcqp power flow
function qcqp_matrices(Yb::SparseMatrixCSC,E::SparseMatrixCSC,network_data::Dict,YLft::SparseMatrixCSC,YLtf::SparseMatrixCSC,YSftE1::SparseMatrixCSC,YStfE2::SparseMatrixCSC)
      nL = size(E,1)
      nB = size(E,2)

      # Build PQ Injection Matrices and Voltage Magnitudes
      Mp  = Dict()
      Mq  = Dict()
      MV2 = Dict()
      for ii = 1:nB
            ek       = spzeros(Complex{Float64},nB,1)
            ek[ii]   = 1
            ev       = spzeros(nB,1)
            ev[ii]   = 1
            zm       = spzeros(nB,nB)
            Yk       = spzeros(Complex{Float64},nB,nB)
            Yk[ii,:] = transpose(Yb[ii,:])
            Mp[ii] = 0.5*[real(Yk + transpose(Yk)) imag(transpose(Yk) - Yk);
                          imag(Yk - transpose(Yk)) real(Yk + transpose(Yk))]

            Mq[ii] = -0.5*[imag(Yk + transpose(Yk)) real(Yk - transpose(Yk));
                           real(transpose(Yk) - Yk) imag(Yk + transpose(Yk))]
            MV2[ii] = [ev*ev' zm;
                       zm    ev*ev']
            
            # Make sure small elements are 0
            Mq[ii][abs.(Mq[ii]) .< 1e-9] .= 0
            Mp[ii][abs.(Mp[ii]) .< 1e-9] .= 0
            MV2[ii][abs.(MV2[ii]) .< 1e-9] .= 0

            # Now, make symmetric and sparse
            Mp[ii]  = sparse(Symmetric(Mp[ii]))
            Mq[ii]  = sparse(Symmetric(Mq[ii]))
            MV2[ii] = sparse(Symmetric(MV2[ii]))
      end

      # Build PQ Flow Matrices and Current Magnitude Flow
      Mpft = Dict()
      Mqft = Dict()
      Mptf = Dict()
      Mqtf = Dict()
      MIft2 = Dict()
      MItf2 = Dict()
      YftM  = YLft + YSftE1
      YtfM  = YLtf + YStfE2
      YMft = [real(YftM) -imag(YftM);
                  imag(YftM)  real(YftM)]
      YMtf = [real(YtfM) -imag(YtfM);
                  imag(YtfM)  real(YtfM)]

      for ii = 1:nL
            fi = findall(x->x==1,E[ii,:])
            ti = findall(x->x==-1,E[ii,:])
            f_bus = fi[1]
            t_bus = ti[1]
            et    = spzeros(Complex{Float64},nB,1)
            ef    = spzeros(Complex{Float64},nB,1)
            et[t_bus] = 1
            ef[f_bus] = 1

            # Define series and shunts
            yft = -Yb[f_bus,t_bus]
            ytf = -Yb[t_bus,f_bus]
            if yft != ytf
                  print("***Error - yft != ytf***")
            end
            yff = network_data["branch"][string(ii)]["g_fr"] + im*network_data["branch"][string(ii)]["b_fr"]
            ytt = network_data["branch"][string(ii)]["g_to"] + im*network_data["branch"][string(ii)]["b_to"]

            # Build matrices
            Yft = ef*(ef')*(yft + yff) - ef*(et')*yft
            Ytf = et*(et')*(ytf + ytt) - et*(ef')*ytf
            Mpft[ii] = 0.5*[real(Yft + transpose(Yft)) imag(transpose(Yft) - Yft);
                            imag(Yft - transpose(Yft)) real(Yft + transpose(Yft))]
            Mptf[ii] = 0.5*[real(Ytf + transpose(Ytf)) imag(transpose(Ytf) - Ytf);
                            imag(Ytf - transpose(Ytf)) real(Ytf + transpose(Ytf))]
            Mqft[ii] = -0.5*[imag(Yft + transpose(Yft)) real(Yft - transpose(Yft));
                             real(transpose(Yft) - Yft) imag(Yft + transpose(Yft))]
            Mqtf[ii] = -0.5*[imag(Ytf + transpose(Ytf)) real(Ytf - transpose(Ytf));
                             real(transpose(Ytf) - Ytf) imag(Ytf + transpose(Ytf))]
            ML = spzeros(2*nL,2*nL)
            ML[ii,ii]       = 1
            ML[ii+nL,ii+nL] = 1
            MIft2[ii] = YMft'*ML*YMft
            MItf2[ii] = YMtf'*ML*YMtf

            # Make sure small elements are 0
            Mpft[ii][abs.(Mpft[ii]) .< 1e-9]   .= 0
            Mqft[ii][abs.(Mqft[ii]) .< 1e-9]   .= 0
            Mptf[ii][abs.(Mptf[ii]) .< 1e-9]   .= 0
            Mqtf[ii][abs.(Mqtf[ii]) .< 1e-9]   .= 0
            MIft2[ii][abs.(MIft2[ii]) .< 1e-9] .= 0
            MItf2[ii][abs.(MItf2[ii]) .< 1e-9] .= 0

            # Now, make symmetric and sparse
            Mpft[ii]  = sparse(Symmetric(Mpft[ii]))
            Mqft[ii]  = sparse(Symmetric(Mqft[ii]))
            Mptf[ii]  = sparse(Symmetric(Mptf[ii]))
            Mqtf[ii]  = sparse(Symmetric(Mqtf[ii]))
            MIft2[ii] = sparse(Symmetric(MIft2[ii]))
            MItf2[ii] = sparse(Symmetric(MItf2[ii]))
      end

      return Mp, Mq, MV2, Mpft, Mptf, Mqft, Mqtf, MIft2, MItf2
end

# Build V^2 Jacobian
function Jacobian_VrVi_to_V2(Vc)
      Vr  = real(Vc)
      Vi  = imag(Vc)
      Mr  = 2*Diagonal(Vr)
      Mi  = 2*Diagonal(Vi)
      Jac = hcat(Mr,Mi)

      # Output
      return Jac
end

# Build PQ Jacobian
function Jacobian_VrVi_to_PQ(Theta,Vc,Ic,NY,Polar)

      # Compute Currents and Voltages
      Ir  = real(Ic)
      Ii  = imag(Ic)
      cTh = cos.(Theta)
      sTh = sin.(Theta)
      Vr  = real(Vc)
      Vi  = imag(Vc)

      # Populate MI
      MIr = Diagonal(Ir)
      MIi = Diagonal(Ii)
      MI  = [MIr MIi;
            -MIi MIr]

      # Populate MV
      MVr = Diagonal(Vr)
      MVi = Diagonal(Vi)
      MV  = [MVr -MVi;
             MVi  MVr]

      # Polar (V,Theta) or Cartesian (Vr,Vi) Jacobian?
      if Polar == true
            # Populate RV
            RV = [Diagonal(cTh) Diagonal(-Vi); Diagonal(sTh) Diagonal(Vr)];

            # Build Jacobian
            Jac = (MI + MV*NY)*RV
      else
            # Build Jacobian
            Jac = (MI + MV*NY)
      end

      # Output
      return Jac
end

# Build I^2 Jacobian
function Jacobian_VrVi_to_I2(Vc,YLE,YSEx,NYIm)

      # Compute Currents and Voltages
      Icf = (YLE + YSEx)*Vc
      Ir  = real(Icf)
      Ii  = imag(Icf)

      # Build Jacobian
      JIcVrVi = NYIm
      nB      = length(Vc)
      nF      = length(Icf)

      # Build apparent power Jacobian
      J = [2*Diagonal(Ir)*JIcVrVi[1:nF,1:nB]+2*Diagonal(Ii)*JIcVrVi[nF+1:end,1:nB] 2*Diagonal(Ir)*JIcVrVi[1:nF,nB+1:end]+2*Diagonal(Ii)*JIcVrVi[nF+1:end,nB+1:end]]

      return J
end