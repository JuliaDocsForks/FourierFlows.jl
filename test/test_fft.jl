function create_testfuncs(g::OneDGrid)
  g.nx > 8 || error("nx = $(g.nx); nx must be > 8")

  T = eltype(g.x)
  m = 5
  k0 = g.k[2] # Fundamental wavenumber

  # Test function
  φ = π/3
  f1 = @. cos(m*k0*g.x + φ)
  f1h = fft(f1)
  f1hr = rfft(f1)

  f1hr_mul = zeros(Complex{T}, g.nkr)
  mul!(f1hr_mul, g.rfftplan, f1)

  # Theoretical values of the fft and rfft
  f1h_th = zeros(Complex{T}, size(f1h))
  f1hr_th = zeros(Complex{T}, size(f1hr))

  for i in 1:g.nk
    if abs(real(g.k[i])) == m*k0
      f1h_th[i] = -exp(sign(real(g.k[i]))*im*φ)*g.nx/2
    end
  end

  for i in 1:g.nkr
    if abs(real(g.k[i]))==m*k0
      f1hr_th[i] = -exp(sign(real(g.kr[i]))*im*φ)*g.nx/2
    end
  end

  f1, f1h, f1hr, f1hr_mul, f1h_th, f1hr_th
end

function create_testfuncs(g::TwoDGrid)
  g.nx > 8 || error("nx = $(g.nx); nx must be > 8")

  T = eltype(g.x)
  m, n = 5, 2
  k0 = g.k[2] # fundamental wavenumber
  l0 = g.l[2] # fundamental wavenumber

  # some test functions
  f1 = @. cos(m*k0*g.x) * cos(n*l0*g.y)
  f2 = @. sin(m*k0*g.x + n*l0*g.y)

  f1h = fft(f1)
  f2h = fft(f2)
  f1hr = rfft(f1)
  f2hr = rfft(f2)

  f1hr_mul = zeros(Complex{T}, (g.nkr, g.nl))
  f2hr_mul = zeros(Complex{T}, (g.nkr, g.nl))
  mul!(f1hr_mul, g.rfftplan, f1)
  mul!(f2hr_mul, g.rfftplan, f2)

  # Theoretical results
  f1h_th = zeros(Complex{T}, size(f1h))
  f2h_th = zeros(Complex{T}, size(f2h))
  f1hr_th = zeros(Complex{T}, size(f1hr))
  f2hr_th = zeros(Complex{T}, size(f2hr))

  for j in 1:g.nl, i in 1:g.nk

    if abs(g.k[i]) == m*k0 && abs(g.l[j]) == n*l0
      f1h_th[i, j] = - g.nx*g.ny/4
    end

    if g.k[i] == m*k0 && g.l[j] == n*l0
      f2h_th[i, j] = -g.nx*g.ny/2
    elseif g.k[i] == -m*k0 && g.l[j] == -n*l0
      f2h_th[i, j] = g.nx*g.ny/2
    end

  end

  f2h_th = -im*f2h_th

  for j in 1:g.nl, i in 1:g.nkr
    if ( abs(g.kr[i]) == m*k0 && abs(g.l[j])==n*l0 )
      f1hr_th[i, j] = -g.nx*g.ny/4
    end
    if ( real(g.kr[i]) == m*k0 && real(g.l[j]) == n*l0 )
      f2hr_th[i, j] = -g.nx*g.ny/2
    elseif ( real(g.kr[i]) == -m*k0 && real(g.l[j]) == -n*l0 )
      f2hr_th[i, j] = g.nx*g.ny/2
    end
  end

  f2hr_th = -im*f2hr_th

  f1, f2, f1h, f2h, f1hr, f2hr, f1hr_mul, f2hr_mul, f1h_th, f1hr_th, f2h_th, f2hr_th
end

# -----------------------------------------------------------------------------
# FFT's TEST FUNCTIONS

function test_fft_cosmx(g::OneDGrid)
    f1, f1h, f1hr, f1hr_mul, f1h_th, f1hr_th = create_testfuncs(g)
    isapprox(f1h, f1h_th, rtol=rtol_fft)
end

function test_rfft_cosmx(g::OneDGrid)
    f1, f1h, f1hr, f1hr_mul, f1h_th, f1hr_th = create_testfuncs(g)
    isapprox(f1hr, f1hr_th, rtol=rtol_fft)
end

function test_rfft_mul_cosmx(g::OneDGrid)
    f1, f1h, f1hr, f1hr_mul, f1h_th, f1hr_th = create_testfuncs(g)
    isapprox(f1hr_mul, f1hr_th, rtol=rtol_fft)
end

function test_fft_cosmxcosny(g::TwoDGrid)
    f1, f2, f1h, f2h, f1hr, f2hr, f1hr_mul, f2hr_mul,
            f1h_th, f1hr_th, f2h_th, f2hr_th = create_testfuncs(g)
    isapprox(f1h, f1h_th, rtol=rtol_fft)
end

function test_rfft_cosmxcosny(g::TwoDGrid)
    f1, f2, f1h, f2h, f1hr, f2hr, f1hr_mul, f2hr_mul,
            f1h_th, f1hr_th, f2h_th, f2hr_th = create_testfuncs(g)
    isapprox(f1hr, f1hr_th, rtol=rtol_fft)
end

function test_rfft_mul_cosmxcosny(g::TwoDGrid)
    f1, f2, f1h, f2h, f1hr, f2hr, f1hr_mul, f2hr_mul,
            f1h_th, f1hr_th, f2h_th, f2hr_th = create_testfuncs(g)
    isapprox(f1hr_mul, f1hr_th, rtol=rtol_fft)
end

function test_fft_sinmxny(g::TwoDGrid)
    f1, f2, f1h, f2h, f1hr, f2hr, f1hr_mulPk, f2hr_mul,
            f1h_th, f1hr_th, f2h_th, f2hr_th = create_testfuncs(g)
    isapprox(f2h, f2h_th, rtol=rtol_fft)
end

function test_rfft_sinmxny(g::TwoDGrid)
    f1, f2, f1h, f2h, f1hr, f2hr, f1hr_mul, f2hr_mul,
            f1h_th, f1hr_th, f2h_th, f2hr_th = create_testfuncs(g)
    isapprox(f2hr, f2hr_th, rtol=rtol_fft)
end

function test_rfft_mul_sinmxny(g::TwoDGrid)
    f1, f2, f1h, f2h, f1hr, f2hr, f1hr_mul, f2hr_mul,
            f1h_th, f1hr_th, f2h_th, f2hr_th = create_testfuncs(g)
    isapprox(f2hr_mul, f2hr_th, rtol=rtol_fft)
end
