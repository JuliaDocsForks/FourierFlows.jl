module KuramotoSivashinsky

export 
  InitialValueProblem, 
  updatevars!, 
  set_u!

using 
  FFTW,
  Reexport

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!

"""
    Problem(; parameters...)

Construct a Kuramoto-Sivashinky problem that solves the equation

∂t u + ∂ₓ⁴u + ∂ₓ²u + u ∂ₓu = 0.
"""
function Problem(;
       nx = 256,
       Lx = 2π,
       dt = 0.01,
  stepper = "RK4")

  g  = OneDGrid(nx, Lx)
  pr = Params()
  vs = Vars(g)
  eq = Equation(pr, g)
  ts = TimeStepper(stepper, dt, eq.LC, g)

  FourierFlows.Problem(g, vs, pr, eq, ts)
end

# Placeholder Params type
struct Params <: AbstractParams; end

"Returns the Equation type for Kuramoto-Sivashinsky."
function Equation(p, g)
  LC = @. g.kr^2 - g.kr^4
  FourierFlows.Equation(LC, calcN!)
end

# Construct Vars type
physicalvars = [:u, :ux, :uux]
transformvars = [:uh, :uxh, :uuxh]
eval(FourierFlows.structvarsexpr(:Vars, physicalvars, transformvars; vardims=1))

"Returns the Vars object for Kuramoto-Sivashinsky."
function Vars(g)
  @createarrays Float64 (g.nx,) u ux uux
  @createarrays Complex{Float64} (g.nkr,) uh uxh uuxh
  Vars(u, ux, uux, uh, uxh, uuxh)
end

"Calculates N = - u uₓ, the nonlinear term for the Kuramoto-Sivashinsky equation."
function calcN!(N, sol, t, s, v, p, g)
  @. v.uh = sol
  @. v.uxh = im*g.kr*sol
  ldiv!(v.u, g.rfftplan, v.uh)
  ldiv!(v.ux, g.rfftplan, v.uxh)
  @. v.uux = v.u*v.ux
  mul!(v.uuxh, g.rfftplan, v.uux)
  @. N = -v.uuxh
  dealias!(N, g)
  nothing
end

# Helper functions
"""
    updatevars!(v, s, g)

Update the vars in v on the grid g with the solution in s.sol.
"""
function updatevars!(v, s, g)
  v.uh .= s.sol
  ldiv!(v.u, g.rfftplan, v.uh)
  nothing
end

updatevars!(prob::AbstractProblem) = updatevars!(prob.vars, prob.state, prob.grid)

"""
    set_u!(prob, u)
    set_u!(s, v, g, u)

Set the solution prob.state.sol as the transform of u and update variables.
"""
function set_u!(s, v, g, u)
  mul!(s.sol, g.rfftplan, u)
  updatevars!(v, s, g)
  nothing
end

set_u!(prob::AbstractProblem, u) = set_u!(prob.state, prob.vars, prob.grid, u)

end # module
