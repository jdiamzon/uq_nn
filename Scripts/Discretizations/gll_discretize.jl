module GllDiscretize

export Discretization, generate_gll_disc

using Base.Threads
using LinearAlgebra

struct Discretization{T}
  nodes::Vector{T}
  weights::Vector{T}
  D::Matrix{T}
end

function generate_gll_disc(n_deg::Int, a::Real, b::Real)
  # ================================================ #
  # ================================================ #
  #=
  # Adapted from Daniele's matlab code
  # This function implements a discretization based on
  # the Gauss-Legendre-Lobatto collocation method
  # Computes the Gauss-Legendre-Lobatto (GLL) N_nodes,
  # weights, and diff matrix in the interval [a,b]
  # Inputs:
  #   n_deg::Int -> polynomial order
  #   (a,b)::Tuple -> interval [a::Real,b::Real]
  # Outputs:
  #   x_grid::Vector -> (n_deg+1) GLL discretization nodes
  #   weights::Vector -> Gauss-Lobatto quadrature weights
  #   D::Matrix -> GLL Differentiation matrix.
  =#
  # ================================================ #
  # ================================================ #

  #= Adding +1 for the coefficient part of the polynomial
  We're adding 1 to account for the constraint from the
  constant term in the orthogonal interpolating polynomial,
  for a total of n_deg+1 constrants. =#
  N_nodes::Int = n_deg + 1;

  # Chebyshev-Gauss-Lobatto N_nodes as our starting guess
  x_cos::Vector = cos.((π/n_deg)*collect(0:n_deg));

  x_u::Vector = collect(range(-1,1,N_nodes));

  # Make a close first guess to reduce iterations
  # in the Newton-Raphson root-finding method
  # used below:
  if n_deg < 3
    x_grid = x_cos;
  else
    x_grid = x_cos + sin.(π*x_u)/(4.0*n_deg);
  end

  # P -> Legendre Vandermonde matrix
  # D -> Diff Matrix
  P::Matrix = zeros(Float64, N_nodes, N_nodes);
  D::Matrix = copy(P);
  weights::Vector = zeros(N_nodes);

  # ================================================ #
  # Computes P(N_nodes) using the (Bonnet) recursion relation
  # First by computing the first and second derivatives,
  # then updates x_grid via the Newton-Raphson method.
  # ------------------------------------------------
  #   Note: keep in mind, other formulas on the internet may use the
  #   recursion variable n = n_deg, whereas we're using the
  #   recursion variable n = N_nodes = n_deg + 1 -> n_deg = N_nodes - 1
  #   c.f. https://en.wikipedia.org/wiki/Legendre_polynomials#Definition_via_generating_function
  #        https://tobydriscoll.net/fnc-julia/globalapprox/orthogonal.html?highlight=legendre#legendre-polynomials
  # ------------------------------------------------
  # ================================================ #
  x_old::Vector = 2.0*ones(size(x_grid));
  # while max(abs.(x_grid-x_old)...) > eps()
  while norm(x_grid - x_old, Inf) > eps()

    x_old = x_grid;

    P[:, 1] = 1.0*ones(N_nodes);
    P[:, 2] = x_grid;

    for k ∈ 2:n_deg
      P[:,k+1] = ((2.0*k - 1.0)*x_grid.*P[:,k] - (k - 1.0)*P[:,k-1])./(1.0*k);
    end

    x_grid = x_old - ( x_grid .* P[:,N_nodes] - P[:,n_deg])./(N_nodes * P[:, N_nodes]);

  end

  # Making N_nodes times copies of our x_grid
  # to compute the differentiation matrix.
  # This could also be done (probably more easily,
  # and efficiently), by indexing and for-loops:
  X = repeat(x_grid,1,N_nodes);
  Xdiff = X - X' + I(N_nodes); # xᵢ - xⱼ, and xᵢ = xⱼ -> 1.0

  # ================================================ #
  # Here are the rules to repeat the GLL differentiation matrix:
  # Dᵢⱼ⁽¹⁾ =
  # { -n_deg(n_deg - 1)/4,    i = j = 0;
  # | Lₙ(xᵢ)/((x_grid - xᵢ)Lₙ(xⱼ)),  i ≠ j;
  # | 0,                      i = j (not 0 or n_deg);
  # { n_deg(n_deg + 1)/4,     i = j = n_deg;
  #
  # where n_deg = degree of orthogonal polynomial (Legendre in this case)
  # ================================================ #

  # Preparing N_nodes times copies of highest degree term
  # of Legendre polynomial, Lₙ(xᵢ) = P[:, N_nodes], evaluated on x_grid xᵢ:
  L = repeat(P[:,N_nodes],1,N_nodes);
  L[diagind(L)] .=1; # L[1:(N_nodes+1):N_nodes*N_nodes] .= 1;
  D=(L./(Xdiff.*L'));
  D[diagind(D)] .=0; # D[1:(N_nodes+1):N_nodes*N_nodes] .= 0;
  D[1] = (N_nodes*n_deg)/4.0;
  D[N_nodes*N_nodes] = -(N_nodes*n_deg)/4.0;

  # ================================================ #
  # Here's the weights
  #
  # c.f. https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Lobatto_rules
  #      https://mathworld.wolfram.com/LobattoQuadrature.html
  #
  # ================================================ #

  weights= 2.0./(n_deg*N_nodes*P[:,N_nodes].^2);

  # aligning our x_grid, weights, and diff matrix D
  # correctly:
  x_grid     = reverse(x_grid);
  weights    = reverse(weights);
  D          = -D;

  # Adjusting our x_grid, weights, and diff matrix D
  # so that they are evaluted correctly on
  # an arbitrary interval:
  D       = (2.0/(b-a))*D;
  weights = ((b-a)/2.0)*weights;
  x_grid  = ((b-a)/2.0)*x_grid .+ (b+a)/2.0;

  gll_disc::Discretization = Discretization(x_grid, weights, D);

  # return x_grid, weights, D
  return gll_disc
end

end