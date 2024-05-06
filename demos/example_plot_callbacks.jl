# set up my model
model = Model(Gurobi.Optimizer)

# add variables and constraints...
@variable(model, x[1:n])
@constraint(model, A*x .<= b)

# ask gurobi to solve with the following callback function:
MOI = MathOptInterface
MOI.set(model, Gurobi.CallbackFunction(), callback_log_plot)

# optimize the model
optimize!(model)

# callback function:
function callback_log_plot(cb_data, cb_where::Cint)
    # only save the data when MIP = 3 -- https://www.gurobi.com/documentation/10.0/refman/cb_codes.html
    MIP = 3
    if cb_where == MIP
          runtimeP = Ref{Cdouble}()
          objbstP  = Ref{Cdouble}()
          objbndP  = Ref{Cdouble}()
          GRBcbget(cb_data, cb_where, GRB_CB_RUNTIME, runtimeP)
          GRBcbget(cb_data, cb_where, GRB_CB_MIP_OBJBST, objbstP)
          GRBcbget(cb_data, cb_where, GRB_CB_MIP_OBJBND, objbndP)
          best  = objbstP[]
          bound = objbndP[]

          # push data to save it
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
    end
    return
end
