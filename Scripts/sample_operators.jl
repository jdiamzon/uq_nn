# using FastGaussQuadrature, Polynomials, SpecialPolynomials
# using Base.Threads
# using LinearAlgebra
include("Discretizations/gll_discretize.jl")
using .GllDiscretize: Discretization

function ℒ_localLin(fᵢ, y, disc::Discretization)
  # ================================================ #
  # Inputs:
  #   f::Vector -> Vector of input function's evaluated points, [f(x₁) f(x₂) …]
  #   y::Vector -> Codomain grid
  #   disc::Discretization -> (wᵢ::Vector -> Quadrature weights,
  #                            Dₓ::Matrix -> Differentiation Matrix)
  # Output:
  #   𝒩_f:: Vector -> Vector of output function's evaluated points, [𝒩_f(x₁) 𝒩_f(x₂) …]
  # ================================================ #
  x = disc.nodes;
  Npoints_y = length(y);
  output_grid_weight = cos.(x)*ones(1, Npoints_y);
  wᵢ = disc.weights;
  Dₓ = disc.D;

  𝒩_f = (fᵢ + (Dₓ * fᵢ))
  return 𝒩_f
end

function ℒ_nonlocalLin(fᵢ, y, disc::Discretization)
  # ================================================ #
  # Inputs:
  #   f::Vector -> Vector of input function's evaluated points, [f(x₁) f(x₂) …]
  #   y::Vector -> Codomain grid
  #   disc::Discretization -> (wᵢ::Vector -> Quadrature weights,
  #                            Dₓ::Matrix -> Differentiation Matrix)
  # Output:
  #   𝒩_f:: Vector -> Vector of output function's evaluated points, [𝒩_f(x₁) 𝒩_f(x₂) …]
  # ================================================ #
  x = disc.nodes;
  Npoints_y = length(y);
  output_grid_weight = cos.(x)*ones(1, Npoints_y);
  wᵢ = disc.weights;
  Dₓ = disc.D;
  α = 0.5;

  # 𝒩_f = (fᵢ.*fᵢ - α*fᵢ.*(Dₓ * fᵢ))
  𝒩_f = fᵢ .* y + fᵢ.*(Dₓ * fᵢ)
  return 𝒩_f
end

function ℒ_lin(fᵢ, y, disc::Discretization)
  # ================================================ #
  # Inputs:
  #   f::Vector -> Vector of input function's evaluated points, [f(x₁) f(x₂) …]
  #   y::Vector -> Codomain grid
  #   disc::Discretization -> (wᵢ::Vector -> Quadrature weights,
  #                            Dₓ::Matrix -> Differentiation Matrix)
  # Output:
  #   𝒩_f:: Vector -> Vector of output function's evaluated points, [𝒩_f(x₁) 𝒩_f(x₂) …]
  # ================================================ #
  x = disc.nodes;
  Npoints_y = length(y);
  output_grid_weight = cos.(x)*ones(1, Npoints_y);
  wᵢ = disc.weights;
  Dₓ = disc.D;

  𝒩_f = wᵢ' * (fᵢ * y' + (Dₓ * fᵢ) .* sin.(π * (y').^2) .* output_grid_weight)
  return 𝒩_f
end

function ℒ_nonlin(fᵢ, y, disc::Discretization)
  # ================================================ #
  # Inputs:
  #   f::Vector -> Vector of input function's evaluated points, [f(x₁) f(x₂) …]
  #   y::Vector -> Codomain grid
  #   disc::Discretization -> (wᵢ::Vector -> Quadrature weights,
  #                            Dₓ::Matrix -> Differentiation Matrix)
  # Output:
  #   𝒩_f:: Vector -> Vector of output function's evaluated points, [𝒩_f(x₁) 𝒩_f(x₂) …]
  # ================================================ #
  x = disc.nodes;
  Npoints_y = length(y);
  output_grid_weight = cos.(x)*ones(1, Npoints_y);
  wᵢ = disc.weights;
  Dₓ = disc.D;

  𝒩_f = wᵢ' * (fᵢ * y' + fᵢ.*(Dₓ * fᵢ) .* sin.(π * (y').^2) .* output_grid_weight)
  return 𝒩_f
end

function ℒ_nonlin2(fᵢ, y, disc::Discretization)
  # ================================================ #
  # Inputs:
  #   f::Vector -> Vector of input function's evaluated points, [f(x₁) f(x₂) …]
  #   y::Vector -> Codomain grid
  #   disc::Discretization -> (wᵢ::Vector -> Quadrature weights,
  #                            Dₓ::Matrix -> Differentiation Matrix)
  # Output:
  #   𝒩_f:: Vector -> Vector of output function's evaluated points, [𝒩_f(x₁) 𝒩_f(x₂) …]
  # ================================================ #
  x = disc.nodes;
  Npoints_y = length(y);
  output_grid_weight = cos.(x)*ones(1, Npoints_y);
  wᵢ = disc.weights;
  Dₓ = disc.D;

  𝒩_f = wᵢ' * (cos.(fᵢ) * y' + (Dₓ * fᵢ) .* sin.(π * (y').^2) .* output_grid_weight)
  return 𝒩_f
end

function ℒ_nonlin3(fᵢ, y, disc::Discretization)
  # ================================================ #
  # Inputs:
  #   f::Vector -> Vector of input function's evaluated points, [f(x₁) f(x₂) …]
  #   y::Vector -> Codomain grid
  #   disc::Discretization -> (wᵢ::Vector -> Quadrature weights,
  #                            Dₓ::Matrix -> Differentiation Matrix)
  # Output:
  #   𝒩_f:: Vector -> Vector of output function's evaluated points, [𝒩_f(x₁) 𝒩_f(x₂) …]
  # ================================================ #
  x = disc.nodes;
  Npoints_y = length(y);
  output_grid_weight = cos.(x)*ones(1, Npoints_y);
  wᵢ = disc.weights;
  Dₓ = disc.D;

  𝒩_f = fᵢ' + wᵢ' * (exp.(fᵢ) * y' + fᵢ.*(Dₓ * fᵢ) .* sin.(π * (y').^2) .* output_grid_weight)
  return 𝒩_f
end


function ℒ_nonlin4(fᵢ, y, disc::Discretization)
  # ================================================ #
  # Inputs:
  #   f::Vector -> Vector of input function's evaluated points, [f(x₁) f(x₂) …]
  #   y::Vector -> Codomain grid
  #   disc::Discretization -> (wᵢ::Vector -> Quadrature weights,
  #                            Dₓ::Matrix -> Differentiation Matrix)
  # Output:
  #   𝒩_f:: Vector -> Vector of output function's evaluated points, [𝒩_f(x₁) 𝒩_f(x₂) …]
  # ================================================ #
  x = disc.nodes;
  Npoints_y = length(y);
  output_grid_weight = cos.(x)*ones(1, Npoints_y);
  wᵢ = disc.weights;
  Dₓ = disc.D;

  𝒩_f = fᵢ + wᵢ' * (exp.(sin.(fᵢ)*(cos.(fᵢ))) + fᵢ * y' + (Dₓ * fᵢ) .* sin.(π * (y').^2) .* output_grid_weight)
  return 𝒩_f
end

function ℒ_nonlin5(fᵢ, y, disc::Discretization)
  # ================================================ #
  # Inputs:
  #   f::Vector -> Vector of input function's evaluated points, [f(x₁) f(x₂) …]
  #   y::Vector -> Codomain grid
  #   disc::Discretization -> (wᵢ::Vector -> Quadrature weights,
  #                            Dₓ::Matrix -> Differentiation Matrix)
  # Output:
  #   𝒩_f:: Vector -> Vector of output function's evaluated points, [𝒩_f(x₁) 𝒩_f(x₂) …]
  # ================================================ #
  x = disc.nodes;
  Npoints_y = length(y);
  output_grid_weight = cos.(x)*ones(1, Npoints_y);
  wᵢ = disc.weights;
  Dₓ = disc.D;

  𝒩_f = fᵢ*fᵢ + wᵢ' * (exp.(sin.(fᵢ)) + fᵢ * y' + ((Dₓ * fᵢ).^2) .* sin.(π * (y').^2) .* output_grid_weight)
  return 𝒩_f
end

## Older sample operators I've tried out ##

# function sample_operator(Ngrids_x::Int, Ngrids_y::Int, func_in::Function)
#   NSAMP = 100000;
#   Nx = 30;
#   # Ny = 20;
#   Ny = 30;

#   x,wx,Dx = generate_nodes_weights_diff(Nx, -1.0, 1.0);
#   y,~,~ = generate_nodes_weights_diff(Ny, -1.0, 1.0);

#   INPUT = zeros(NSAMP, Nx+1);
#   OUTPUT = zeros(NSAMP,Ny+1);

#   # common_dom = collect(range(-1.0, 1.0, 1000));

#   # Threads.@threads for ii in 1:NSAMP
#   for ii in 1:NSAMP
#     # f = randn(Nx+1,1); # Function values at collocation points
#     # f = sin.(x); # Function values at collocation points

#     weight_x = cos.(x)*ones(1, Ny+1);

#     INPUT[ii,:] = f;
#     OUTPUT[ii,:] = wx'*(f*y' + (Dx*f)*sin.(π*(y').^2).*weight_x);
#   end

#   return x, Nx, y, Ny, INPUT, OUTPUT

# end

# function N_s(nodes, D, y, f::Function)
#   # s = [cos(z) for z in nodes]
#   s = [f(z) for z in nodes]
#   kernel = exp.(-s*y' - (D*s)*(y.^2)')
#   integral = kernel ⋅ [w for w in weights]
#   return integral
# end

# function N_s_pointwise(nodes,D, y, s, f::Function)
#   # s = [f(z) for z in nodes]
#   kernel = exp.(-s.*y - (D*s)*(y.^2))
#   integral = kernel ⋅ [w for w in weights]
#   return integral
# end

# function sample_operator_expIn(input_function::Function)
#   NSAMP = 100000;
#   Ngrids_x = 30;
#   # Ngrids_y = 20;
#   Ngrids_y = 30;

#   x,wx,Dx = generate_nodes_weights_diff(Ngrids_x, -1.0, 1.0);
#   y,~,~ = generate_nodes_weights_diff(Ngrids_y, -1.0, 1.0);

#   INPUT = zeros(NSAMP, Ngrids_x+1);
#   OUTPUT = zeros(NSAMP,Ngrids_y+1);

#   # common_dom = collect(range(-1.0, 1.0, 1000));

#   Threads.@threads for ii in 1:NSAMP
#     # f = randn(Ngrids_x+1,1); # Function values at collocation points
#     # f = exp.(x); # Function values at collocation points
#     f = input_function

#     weight_x = cos.(x)*ones(1, Ngrids_y+1);

#     INPUT[ii,:] = f;
#     OUTPUT[ii,:] = wx'*(f*y' + (Dx*f)*sin.(π*(y').^2).*weight_x);
#   end

#   return x, Ngrids_x, y, Ngrids_y, INPUT, OUTPUT

# end


# function generate_data( Ngrids_x::Int,
#                         Ngrids_y::Int,
#                         func_in::Function,
#                         operator::Function,
#                         NSAMP::Int = 10000)
#   # Number of sample input/output curves
#   # NSAMP = 10000;
#   # Ngrids_x = 30;
#   # Ngrids_y = 30;

#   # x,wx,Dx = generate_gll_disc(Ngrids_x, -1.0, 1.0);
#   # y,_,_ = generate_gll_disc(Ngrids_y, -1.0, 1.0);
#   disc_x = generate_gll_disc(Ngrids_x, -1.0, 1.0);
#   disc_y = generate_gll_disc(Ngrids_y, -1.0, 1.0);

#   INPUT = zeros(NSAMP, Ngrids_x+1);
#   OUTPUT = zeros(NSAMP,Ngrids_y+1);

#   # Threads.@threads for ii in 1:NSAMP
#   for ii in 1:NSAMP
#     # # f = randn(Ngrids_x+1,1); # Function values at collocation points
#     # f = randn(length(x));
#     # # f = exp.(x); # Function values at collocation points

#     # weight_x = cos.(x)*ones(1, Ngrids_y+1);

#     # INPUT[ii,:] = f;
#     # # OUTPUT[ii,:] = wx'*(f*y' + (Dx*f)*sin.(π*(y').^2).*weight_x);
#     # OUTPUT[ii,:] = test_operator(f, x, y, wx,Dx, weight_x);
#     f = func_in(x);
#     y = disc_y.nodes;

#     INPUT[ii,:] = f;
#     OUTPUT[ii,:] = operator(f, y, disc_x);
#     # OUTPUT[ii,:] = operator(f, x, y, wx, Dx);
#   end

#   return x, y, INPUT, OUTPUT

# end

##=============================##
# Generate data function that for
# some reason was also included here
##=============================##
# function generate_data( Ngrids_x::Int,
#                         Ngrids_y::Int,
#                         func_in::Function,
#                         operator::Function,
#                         NSAMP::Int = 10000)
#   # Number of sample input/output curves

#   disc_x = generate_gll_disc(Ngrids_x, -1.0, 1.0);
#   disc_y = generate_gll_disc(Ngrids_y, -1.0, 1.0);

#   x = disc_x.nodes;
#   y = disc_y.nodes;
#   INPUT = zeros(Ngrids_x+1, NSAMP);
#   OUTPUT = zeros(Ngrids_y+1,NSAMP);

#   # Threads.@threads for ii in 1:NSAMP
#   for ii in 1:NSAMP
#     f = func_in(x);

#     INPUT[:,ii] = f;
#     OUTPUT[:,ii] = operator(f, y, disc_x);
#   end

#   return x, y, INPUT, OUTPUT
# end