# VerifyStrongSDP

This repository containts code for collecting data, training neural networks, bounding big-M, and running NN performance verification (both sequential targeted tightening (STT) and MIQP) as reported in the following preprint:

*Global Performance Guarantees for Neural Network Models of AC Power Flow*

This paper was submitted to the *IEEE IAS* for signle-blind peer review.

The models are implemented in ```Julia-1.10``` Language, using the ```Flux.jl``` library for machine learning and the ```JuMP.jl``` library for mathematical programming. Optimization problems are solved with ```MOSEK``` (for SDPs), ```Gurobi``` (for MILPs and MIQP), and ```IPOPT``` (for NLPs). Before running the code, make sure to activate the virtual environment from ```Project.toml```, e.g., by running 

```
julia> ]
(@v1.10) pkg> activate .
(VerifyStrongSDP) pkg> instantiate
```

For any questions, please constact schevali@uvm.edu.
