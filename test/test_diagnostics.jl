module Testmodule

using FourierFlows, FFTW
import LinearAlgebra: norm, mul!, ldiv!

export updatevars!, set_u!

struct Params <: AbstractParams
    mu::Float64              # drag
end

function Equation(p, g)
  LC = 0.0*ones(size(g.kr))
  @. LC = -p.mu
  FourierFlows.Equation(LC, calcN!)
end

# Construct Vars type
physvars = [:u]
transvars = [:uh]
expr = FourierFlows.structvarsexpr(:Vars, physvars, transvars; vardims=1)
eval(expr)

"Returns the Vars object for Kuramoto-Sivashinsky."
function Vars(g)
  @createarrays Float64 (g.nx,) u
  @createarrays Complex{Float64} (g.nkr,) uh
  Vars(u, uh)
end

function calcN!(N, sol, t, s, v, p, g)
  nothing
end

function set_u!(prob::AbstractProblem, u)
  v, s, g = prob.vars, prob.state, prob.grid
  mul!(s.sol, g.rfftplan, u)
  updatevars!(v, s, g)
  nothing
end

function updatevars!(v, s, g)
  v.uh .= s.sol
  ldiv!(v.u, g.rfftplan, v.uh)
  nothing
end

updatevars!(prob::AbstractProblem) = updatevars!(prob.vars, prob.state, prob.grid)


@inline function pseudoenergy(prob)
  s = prob.state
  sum(abs.(s.sol).^2)
end

@inline function pseudoenstrophy(prob)
  s = prob.state
  0.5*sum(abs.(s.sol).^2)
end

end #module

function testsimplediagnostics()
    nx = 16
    Lx = 2π
    mu = 0.02
    dt = 0.01

    g = OneDGrid(nx, Lx)
    p = Testmodule.Params(mu)
    v = Testmodule.Vars(g)
    eq = Testmodule.Equation(p, g)
    ts = FourierFlows.ETDRK4TimeStepper(dt, eq.LC)
    prob = FourierFlows.Problem(g, v, p, eq, ts)

    Random.seed!(1234)
     u0 = randn(size(g.x))

    Testmodule.set_u!(prob, u0)

    nsteps = 200
    extrasteps = 20

    freqE = 2
    E = Diagnostic(Testmodule.pseudoenergy, prob; nsteps=nsteps, freq=freqE)

    freqZ = 1
    Z = Diagnostic(Testmodule.pseudoenstrophy, prob; nsteps=nsteps, freq=freqZ)

    diags = [E, Z]


    nstepstot = nsteps + extrasteps
    while prob.step < nstepstot
      stepforward!(prob, diags, 1)
    end

    stepforward!(prob, 1)
    increment!(Z)
    stepforward!(prob, 1)
    increment!([E,Z])


     resize!(Z, Z.count+2)
     Z.count += 1
     resize!(E, E.count+1)
     E.count += 1

    stepforward!(prob, 1)
    update!(Z)
    Z.count += 1
    stepforward!(prob, 1)
    update!([E, Z])
    E.count += 1
    Z.count += 1

    (
     isapprox(length(E.data), 2+Int(round(nstepstot/freqE+1))) &&
     isapprox(length(Z.data), 4+Int(round(nstepstot/freqZ+1))) &&
     isapprox(norm(E.data), norm(E.data[1]*exp.(-2*mu*E.time)), rtol=1e-13) &&
     isapprox(norm(Z.data), norm(Z.data[1]*exp.(-2*mu*Z.time)), rtol=1e-13)
    )
end
