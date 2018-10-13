"""
    ZeroDGrid()

Constructs a placeholder grid object for "0D" problems (in other words, systems of ODEs).
"""
struct ZeroDGrid <: AbstractGrid end

"""
    OneDGrid(nx, Lx; x0=-Lx/2, nthreads=Sys.CPU_THREADS, effort=FFTW.MEASURE)

Constrcut a OneDGrid object with size `Lx`, resolution `nx`, and leftmost
position `x0`. FFT plans are generated for `nthreads` CPUs using
FFTW flag `effort`.
"""
struct OneDGrid{T} <: AbstractOneDGrid
  nx::Int
  nk::Int
  nkr::Int
  Lx::T
  dx::T
  x::Array{T,1}
  k::Array{T,1}
  kr::Array{T,1}
  invksq::Array{T,1}
  invkrsq::Array{T,1}
  fftplan::FFTW.cFFTWPlan{Complex{T},-1,false,1}
  rfftplan::FFTW.rFFTWPlan{T,-1,false,1}
  # Range objects that access the aliased part of the wavenumber range
  kalias::UnitRange{Int}
  kralias::UnitRange{Int}
end

function OneDGrid(nx, Lx; x0=-0.5*Lx, nthreads=Sys.CPU_THREADS, effort=FFTW.MEASURE)
  T = typeof(Lx)
  dx = Lx/nx
  x = Array{T}(range(x0, step=dx, length=nx))

  nk = nx
  nkr = Int(nx/2+1)

  i1 = 0:Int(nx/2)
  i2 = Int(-nx/2+1):-1
  k = Array{T}(2π/Lx*cat(i1, i2; dims = 1))
  kr = Array{T}(2π/Lx*cat(i1; dims = 1))

  invksq = @. 1/k^2
  invksq[1] = 0
  invkrsq = @. 1/kr^2
  invkrsq[1] = 0

  FFTW.set_num_threads(nthreads)
  fftplan   = plan_fft(Array{Complex{T},1}(undef, nx); flags=effort)
  rfftplan  = plan_rfft(Array{T,1}(undef, nx); flags=effort)

  # Index endpoints for aliased i, j wavenumbers
  iaL, iaR = Int(floor(nk/3))+1, 2*Int(ceil(nk/3))-1
  kalias  = iaL:iaR
  kralias = iaL:nkr

  OneDGrid(nx, nk, nkr, Lx, dx, x, k, kr, invksq, invkrsq,
           fftplan, rfftplan, kalias, kralias)
end

"""
    TwoDGrid(nx, Lx)
    TwoDGrid(nx, Lx, ny, Ly; x0=-Lx/2, y0=-Ly/2, nthreads=Sys.CPU_THREADS, effort=FFTW.MEASURE)

Constrcut a TwoDGrid object. The two-dimensional domain has size (Lx, Ly),
resolution (nx, ny) and bottom left corner at (x0, y0). FFT plans are generated
which use nthreads threads with the specified planning effort.
"""
struct TwoDGrid{T} <: AbstractTwoDGrid
  nx::Int
  ny::Int
  nk::Int
  nl::Int
  nkr::Int
  Lx::T
  Ly::T
  dx::T
  dy::T
  x::Array{T,2}
  y::Array{T,2}
  k::Array{T,2}
  l::Array{T,2}
  kr::Array{T,2}
  Ksq::Array{T,2}      # k^2 + l^2
  invKsq::Array{T,2}   # 1/Ksq, invKsq[1, 1]=0
  Krsq::Array{T,2}     # kr^2 + l^2
  invKrsq::Array{T,2}  # 1/Krsq, invKrsq[1, 1]=0

  fftplan::FFTW.cFFTWPlan{Complex{T},-1,false,2}
  rfftplan::FFTW.rFFTWPlan{T,-1,false,2}

  # Range objects that access the aliased part of the wavenumber range
  kalias::UnitRange{Int}
  kralias::UnitRange{Int}
  lalias::UnitRange{Int}
end

function TwoDGrid(nx, Lx, ny=nx, Ly=Lx; x0=-0.5*Lx, y0=-0.5*Ly, nthreads=Sys.CPU_THREADS, effort=FFTW.MEASURE,
                  T=Float64)
  dx = Lx/nx
  dy = Ly/ny
  nk = nx
  nl = ny
  nkr = Int(nx/2+1)

  # Physical grid
  x = Array{T}(reshape(range(x0, step=dx, length=nx), (nx, 1)))
  y = Array{T}(reshape(range(y0, step=dy, length=ny), (1, ny)))

  # Wavenubmer grid
  i1 = 0:Int(nx/2)
  i2 = Int(-nx/2+1):-1
  j1 = 0:Int(ny/2)
  j2 = Int(-ny/2+1):-1

  k  = reshape(2π/Lx*cat(i1, i2, dims=1), (nk, 1))
  kr = reshape(2π/Lx*cat(i1, dims=1), (nkr, 1))
  l  = reshape(2π/Ly*cat(j1, j2, dims=1), (1, nl))

  Ksq  = @. k^2 + l^2
  invKsq = @. 1/Ksq
  invKsq[1, 1] = 0

  Krsq = @. kr^2 + l^2
  invKrsq = @. 1/Krsq
  invKrsq[1, 1] = 0

  # FFT plans
  FFTW.set_num_threads(nthreads)
  fftplan   = plan_fft(Array{Complex{T},2}(undef, nx, ny); flags=effort)
  rfftplan  = plan_rfft(Array{T,2}(undef, nx, ny); flags=effort)

  # Index endpoints for aliased i, j wavenumbers
  iaL, iaR = Int(floor(nk/3))+1, 2*Int(ceil(nk/3))-1
  jaL, jaR = Int(floor(nl/3))+1, 2*Int(ceil(nl/3))-1

  kalias  = iaL:iaR
  kralias = iaL:nkr
  lalias  = jaL:jaR

  TwoDGrid(nx, ny, nk, nl, nkr, Lx, Ly, dx, dy, x, y,
           k, l, kr, Ksq, invKsq, Krsq, invKrsq,
           fftplan, rfftplan, kalias, kralias, lalias)
end

"""
    gridpoints(g)

Returns the collocation points of the grid `g` in 2D arrays `X, Y`.
"""
function gridpoints(g)
  X = [ g.x[i] for i=1:g.nx, j=1:g.ny]
  Y = [ g.y[j] for i=1:g.nx, j=1:g.ny]
  X, Y
end

"""
    dealias!(a, g, kalias)

Dealias `a` on the grid `g` with aliased x-wavenumbers `kalias`.
"""
function dealias!(a, g::OneDGrid)
  kalias = size(a, 1) == g.nkr ? g.kralias : g.kalias
  dealias!(a, g, kalias)
  nothing
end

function dealias!(a, g::OneDGrid, kalias)
  @views @. a[kalias, :] = 0
  nothing
end

function dealias!(a, g::TwoDGrid)
  kalias = size(a, 1) == g.nkr ? g.kralias : g.kalias
  dealias!(a, g, kalias)
  nothing
end

function dealias!(a, g::TwoDGrid, kalias)
  @views @. a[kalias, g.lalias, :] = 0
  nothing
end

"""
    makefilter(K; order=4, innerK=0.65, outerK=1)

Returns a filter acting on the non-dimensional wavenumber K that decays exponentially
for K>innerK, thus removing high-wavenumber content from a spectrum it is multiplied with.
The decay rate is determined by order and outerK determines the outer wavenumber at which
the filter is smaller than machine precision.
"""
function makefilter(K; order=4, innerK=0.65, outerK=1)
  TK = typeof(K)
  K = Array(K)
  decay = 15*log(10) / (outerK-innerK)^order # decay rate for filtering function
  filt = @. exp( -decay*(K-innerK)^order )
  filt[real.(K) .< innerK] .= 1
  TK(filt)
end

function makefilter(g::AbstractTwoDGrid; realvars=true, kwargs...)
  K = realvars ?
      @.(sqrt((g.kr*g.dx/π)^2 + (g.l*g.dy/π)^2)) : @.(sqrt((g.k*g.dx/π)^2 + (g.l*g.dy/π)^2))
  makefilter(K; kwargs...)
end

function makefilter(g::AbstractOneDGrid; realvars=true, kwargs...)
  K = realvars ? g.kr*g.dx/π : @.(abs(g.k*g.dx/π))
  makefilter(K; kwargs...)
end

makefilter(g, T, sz; kwargs...) = ones(T, sz).*makefilter(g; realvars=sz[1]==g.nkr, kwargs...)
