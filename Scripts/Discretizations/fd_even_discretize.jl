using LinearAlgebra

struct Discretization{T}
    nodes::Vector{T}
    weights::Vector{T}
    D::Matrix{T}
end

function generate_fd_disc(N::Int)
    # Generate evenly spaced nodes
    nodes = range(0, 1, length=N+1)
    h = nodes[2] - nodes[1]  # Grid spacing

    # Weights (not used in finite difference, but keeping for consistency)
    weights = ones(N+1) * h

    # Initializing differentiation matrix D for second-order finite difference
    D = zeros(Float64, N+1, N+1)

    # Central difference for interior points
    for i in 2:N
        D[i, i-1] = 1.0
        D[i, i] = -2.0
        D[i, i+1] = 1.0
    end

    # One-sided difference for boundary points
    # Forward difference at the first point
    D[1, 1] = 1.0
    D[1, 2] = -2.0
    D[1, 3] = 1.0

    # Backward difference at the last point
    D[N+1, N-1] = 1.0
    D[N+1, N] = -2.0
    D[N+1, N+1] = 1.0

    # Scale the differentiation matrix by 1/h^2
    D *= 1.0 / h^2

    fd_disc = Discretization(nodes, weights, D)
    return fd_disc
end

# # Example usage
# N = 10  # Number of intervals
# fd_disc = generate_fd_disc(N)

# println("Nodes: ", fd_disc.nodes)
# println("Weights: ", fd_disc.weights)
# println("Differentiation Matrix D: ", fd_disc.D)
