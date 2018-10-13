# Physical grid tests
testdx(g) = abs(sum(g.x[2:end].-g.x[1:end-1]) - (g.nx-1)*g.dx) < rtol_grid*(g.nx-1)
testdy(g) = abs(sum(g.y[2:end].-g.y[1:end-1]) - (g.ny-1)*g.dy) < rtol_grid*(g.nx-1)

testx(g) = isapprox(g.x[end]-g.x[1], g.Lx-g.dx, rtol=rtol_grid)
testy(g) = isapprox(g.y[end]-g.y[1], g.Ly-g.dy, rtol=rtol_grid)

function testX(g) 
  X, Y = gridpoints(g)
  sum(X[end, :].-X[1, :]) - (g.Lx-g.dx)*g.ny < rtol_grid*g.ny
end

function testY(g)
  X, Y = gridpoints(g)
  sum(Y[:, end].-Y[:, 1]) - (g.Ly-g.dy)*g.nx < rtol_grid*g.nx
end

# Test proper arrangement of fft wavenumbers
testk(g) = sum(g.k[2:g.nkr-1] .+ reverse(g.k[g.nkr+1:end], dims=1)) == 0.0
testkr(g) = sum(g.k[1:g.nkr] .- g.kr) == 0.0
testl(g) = sum(g.l[:, 2:Int(g.ny/2)] .+ reverse(g.l[:, Int(g.ny/2+2):end], dims=2)) == 0.0
