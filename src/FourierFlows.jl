module FourierFlows

export
  AbstractProblem,
  AbstractVars,
  AbstractParams,
  AbstractEquation,

  ZeroDGrid,
  OneDGrid,
  TwoDGrid,
  dealias!,
  gridpoints,

  State,
  DualState,
  unpack,

  Diagnostic,
  resize!,
  update!,
  increment!,

  Output,
  saveoutput,
  saveproblem,
  groupsize,
  savediagnostic,

  @createarrays,

  TimeStepper,
  ForwardEulerTimeStepper,
  FilteredForwardEulerTimeStepper,
  RK4TimeStepper,
  FilteredRK4TimeStepper,
  DualRK4TimeStepper,
  DualFilteredRK4TimeStepper,
  ETDRK4TimeStepper,
  FilteredETDRK4TimeStepper,
  DualETDRK4TimeStepper,
  DualFilteredETDRK4TimeStepper,
  AB3TimeStepper,
  FilteredAB3TimeStepper,
  stepforward!

using
  FFTW,
  Statistics,
  JLD2,
  Interpolations

import Base: resize!, getindex, setindex!, push!, append!, fieldnames
import LinearAlgebra: mul!, ldiv!

abstract type AbstractGrid end
abstract type AbstractParams end
abstract type AbstractVars end
abstract type AbstractTimeStepper end
abstract type AbstractEquation end
abstract type AbstractState end
abstract type AbstractProblem end

abstract type AbstractTwoDGrid <: AbstractGrid end
abstract type AbstractOneDGrid <: AbstractGrid end

abstract type AbstractDiagnostic end

# ------------------
# Base functionality
# ------------------

include("problemstate.jl")
include("domains.jl")
include("diagnostics.jl")
include("output.jl")
include("utils.jl")
include("timesteppers.jl")


# -------
# Physics modules
# -------

include("traceradvdiff.jl")
include("kuramotosivashinsky.jl")


end # module
