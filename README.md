# VerifyStrongSDP

This repository containts code for collecting data, training neural networks, bounding big-M, and running NN performance verification (both sequential targeted tightening (STT) and MIQP) as reported in the following preprint:

*Global Performance Guarantees for Neural Network Models of AC Power Flow*

submitted to the *TWPRS* for signle-blind peer review.

The models are implemented in ```Julia-1.8``` Language, using ```JuMP.jl``` using ```Flux.jl``` library for machine learning and JuMP.jl library for mathematical programming. Before running the code, make sure to activate the virtual environment from ```Project.toml``` files stored in each folder, e.g., by running 

```
julia> ]
(@v1.8) pkg> activate .
(VerifyStrongSDP) pkg> instantiate
```

For any questions, please constact schev@dtu.dk.
