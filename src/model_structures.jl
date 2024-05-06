mutable struct ac_power_flow_model_instance
    ref_bus::Int
    num_buses::Int
    num_lines::Int
    bus_Vmag2_QMat::Dict
    bus_Pinj_QMat::Dict
    bus_Qinj_QMat::Dict
    line_pft_QMat::Dict
    line_qft_QMat::Dict
    line_Ift2_QMat::Dict
    line_ptf_QMat::Dict
    line_qtf_QMat::Dict
    line_Itf2_QMat::Dict
    V_max::Vector{Float64}
    V_min::Vector{Float64}
    Ift2_max::Vector{Float64}
    Itf2_max::Vector{Float64}
    Pinj_max::Vector{Float64}
    Pinj_min::Vector{Float64}
    Qinj_max::Vector{Float64}
    Qinj_min::Vector{Float64}
end

mutable struct ac_nn_model_instance
    W0::Matrix{Float64}
    b0::Vector{Float64}
    W1::Matrix{Float64}
    b1::Vector{Float64}
    M_max::Vector{Float64}
    M_min::Vector{Float64}
    In_Mean::Matrix{Float64}
    In_STD::Matrix{Float64}
    Out_Mean::Matrix{Float64}
    Out_STD::Matrix{Float64}
    J0::Matrix{Float64}
    r0::Vector{Float64}
    setzero_threshold::Float64
    accuracy_loss_threshold::Float64
end