using FastGaussQuadrature, Polynomials, SpecialPolynomials
using LinearAlgebra


struct Discretization{T}
  nodes::Vector{T}
  weights::Vector{T}
  D::Matrix{T}
end

function generate_gl_disc(N::Int)
  # N = 128  # Degree of Legendre polynomial{}

  # Compute the roots of the Legendre polynomial P_N, i.e. Gauss-Legendre nodes
  # nodes = gausslegendre(N+1)[1]

  nodes, weights = gausslegendre(N+1)

  # Initializing empty matrix for the differentiation matrix D
  D = zeros(N + 1, N + 1)

  # for i in 0:N
  #   P = basis(Legendre, i)  # Get the i-th Legendre polynomial
  # end
  #P'_{N+1}
  P_N_plus_1_prime = derivative(basis.(Legendre,N + 1))

  D = zeros(Float64, N+1, N+1)
  # Populating differentiation matrix D
  @threads for i in 1:N+1
    for j in 1:N+1
        if i == j
            D[i, j] = nodes[i] / (1 - nodes[i]^2)
        else
            # Off-diagonal elements
            zi = nodes[i]
            zj = nodes[j]
            D[i, j] = P_N_plus_1_prime(zi) / ((zi - zj) * P_N_plus_1_prime(zj))
        end
    end
  end

  gl_disc = Discretization(nodes, weights, D)
  # return nodes, weights, D
  return gl_disc
end

