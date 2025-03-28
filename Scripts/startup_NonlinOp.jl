include("../Scripts/utils/approx_formula.jl")
include("../Scripts/utils/create_model.jl")
include("../Scripts/data_generator.jl")
include("../Scripts/Discretizations/gll_discretize.jl")
include("../Scripts/file_names.jl")
include("../Scripts/noise_generator.jl")
include("../Scripts/sample_operators.jl")
include("../Scripts/utils/plot_data.jl")
include("../Scripts/utils/unpack_params.jl")

import Enzyme, Zygote

using Peaks

using StaticArrays

# using RemoteREPL
# using GLMakie, # for primarily 3D plotting
using CairoMakie, # Primary Plotting library, for vector/svg output and 2D plotting
  PairPlots

using DataFrames,
  DifferentiationInterface,
  DSP,
  # Enzyme,
  # CSV,
  # FiniteDifferences,
  ComponentArrays,
  FourierTools,
  Format,
  JLD2, # Used for saving
  KernelDensity,
  LinearAlgebra,
  Lux,
  Random,
  Reactant,
  # Zygote,
  Optimisers,
  Polynomials,
  Printf,
  Distributions,
  Statistics,
  SpecialFunctions

# Initiate rng:

rng = MersenneTwister()
Random.seed!(rng, 12345)

#=
Generate Training Data:
=#

function test_func(x)
  return randn(length(x))
end
# Nx = Ny = 10;
Nx::Int = 30;
Ny::Int = 30;
# num_samples = 10_000;
num_samples::Int = 100_000;
# ℒ_op = ℒ_nonlin2
ℒ_op::Function = ℒ_nonlin
# ℒ_op::Function = ℒ_nonlin
# ℒ_op = ℒ_localLin

nodes_x, nodes_y, train_input_fns, train_output_fns = generate_data(Nx, Ny, test_func, ℒ_op, num_samples)

#=
Defining the Neural Network:
=#

# The device to load data on.  Can be
# either the CPU or the GPU.
const dev_cpu::CPUDevice = cpu_device();

# Define the neural network
#=
# max_layer_amounts -
# neuron_sizes      - is a list (vector) of # neurons in each (hidden) layer
# 𝛟                 - list of activation functions in each (hidden) layer
=#
max_layer_amounts::Int = 31;
M_neurons::Int = 64;
neuron_sizes::Vector{Int} = fill(M_neurons, max_layer_amounts);
ϕ::Function = leakyrelu;
𝛟::Vector{<:Function} = fill(ϕ, max_layer_amounts);

model_1L::Chain, params_1L::NamedTuple, layer_states_1L::NamedTuple = create_nlayer_mlp(Nx,
  Ny,
  neuron_sizes,
  𝛟; num_hidden_layers=1);

# N-Layers:

model_2L::Chain, params_2L::NamedTuple, layer_states_2L::NamedTuple = create_nlayer_mlp(Nx,
  Ny,
  neuron_sizes,
  𝛟; num_hidden_layers=2);

model_3L::Chain, params_3L::NamedTuple, layer_states_3L::NamedTuple = create_nlayer_mlp(Nx,
  Ny,
  neuron_sizes,
  𝛟; num_hidden_layers=3);

model_5L::Chain, params_5L::NamedTuple, layer_states_5L::NamedTuple = create_nlayer_mlp(Nx,
  Ny,
  neuron_sizes,
  𝛟; num_hidden_layers=5);

model_7L::Chain, params_7L::NamedTuple, layer_states_7L::NamedTuple = create_nlayer_mlp(Nx,
  Ny,
  neuron_sizes,
  𝛟; num_hidden_layers=7);

model_10L::Chain, params_10L::NamedTuple, layer_states_10L::NamedTuple = create_nlayer_mlp(Nx,
  Ny,
  neuron_sizes,
  𝛟; num_hidden_layers=10);

model_12L::Chain, params_12L::NamedTuple, layer_states_12L::NamedTuple = create_nlayer_mlp(Nx,
  Ny,
  neuron_sizes,
  𝛟; num_hidden_layers=12);

model_18L::Chain, params_18L::NamedTuple, layer_states_18L::NamedTuple = create_nlayer_mlp(Nx,
  Ny,
  neuron_sizes,
  𝛟; num_hidden_layers=18);


model_20L::Chain, params_20L::NamedTuple, layer_states_20L::NamedTuple = create_nlayer_mlp(Nx,
  Ny,
  neuron_sizes,
  𝛟; num_hidden_layers=20);

#=
Loading Model if Previously Trained:
(See file_names.jl)
=#

NN_savefile::JLD2.JLDFile = jldopen("../Data/Saved_Models/$(file_1layer["CUDA"]["Nonlin"]).jld2")

# Loading and Storing Parameters from 1-Layer Model:

layer_states_1L::NamedTuple = NN_savefile["layer_states"];
params_1L::NamedTuple = NN_savefile["params"];
𝚯_1L::ComponentArray{<:Float32} = params_1L |> ComponentArray;

W_01L = 𝚯_1L[:layer_1][:weight];
b_01L = 𝚯_1L[:layer_1][:bias];
A1L = 𝚯_1L[:layer_2][:weight];

# Loading and Storing Parameters from N-Layer Model:

NN_savefile_2layer::JLD2.JLDFile = jldopen("../Data/Saved_Models/$(file_2layer["CUDA"]["Nonlin"]).jld2");

NN_savefile_3layer::JLD2.JLDFile = jldopen("../Data/Saved_Models/$(file_3layer["CUDA"]["Nonlin"]).jld2");

NN_savefile_5layer::JLD2.JLDFile = jldopen("../Data/Saved_Models/$(file_5layer["CUDA"]["Nonlin"]).jld2");

NN_savefile_7layer::JLD2.JLDFile = jldopen("../Data/Saved_Models/$(file_7layer["CUDA"]["Nonlin"]).jld2");

NN_savefile_10layer::JLD2.JLDFile = jldopen("../Data/Saved_Models/$(file_10layer["CUDA"]["Nonlin"]).jld2");

NN_savefile_20layer::JLD2.JLDFile = jldopen("../Data/Saved_Models/$(file_20layer["CUDA"]["Nonlin"]).jld2")

layer_states_2L::NamedTuple = NN_savefile_2layer["layer_states"];
params_2L = NN_savefile_2layer["params"];
𝚯_2L::ComponentArray{<:Float32} = params_2L |> ComponentArray

layer_states_3L::NamedTuple = NN_savefile_3layer["layer_states"];
params_3L = NN_savefile_3layer["params"];
𝚯_3L::ComponentArray{<:Float32} = params_3L |> ComponentArray

layer_states_5L::NamedTuple = NN_savefile_5layer["layer_states"];
params_5L = NN_savefile_5layer["params"];
𝚯_5L::ComponentArray{<:Float32} = params_5L |> ComponentArray

layer_states_7L::NamedTuple = NN_savefile_7layer["layer_states"];
params_7L::NamedTuple = NN_savefile_7layer["params"];
𝚯_7L::ComponentArray{<:Float32} = params_7L::NamedTuple |> ComponentArray

layer_states_10L::NamedTuple = NN_savefile_10layer["layer_states"];
params_10L = NN_savefile_10layer["params"];
𝚯_10L::ComponentArray{<:Float32} = params_10L |> ComponentArray

# layer_states_12L::NamedTuple = NN_savefile_12layer["layer_states"];
# params_12L = NN_savefile_12layer["params"];
# 𝚯_12L::ComponentArray{<:Float32} = params_12L |> ComponentArray

# layer_states_18L::NamedTuple = NN_savefile_18layer["layer_states"];
# params_18L = NN_savefile_18layer["params"];
# 𝚯_18L::ComponentArray{<:Float32} = params_18L |> ComponentArray

layer_states_20L::NamedTuple = NN_savefile_20layer["layer_states"];
params_20L = NN_savefile_20layer["params"];
𝚯_20L::ComponentArray{<:Float32} = params_20L |> ComponentArray

# Defining Parameters from Component Arrays into Variables: 1-Layer

𝐖_1L, 𝐛_1L = unpack_params(𝚯_1L);
𝐖_1L::Vector{Matrix{Float32}}
𝐛_1L::Vector{Vector{Float32}}
𝐀_1L::Matrix{Float32} = 𝐖_1L[end];

# Defining Parameters from Component Arrays into Variables: N-Layer

𝐖_2L, 𝐛_2L = unpack_params(𝚯_2L);
𝐖_2L::Vector{Matrix{Float32}}
𝐛_2L::Vector{Vector{Float32}}
𝐀_2L::Matrix{Float32} = 𝐖_2L[end];

𝐖_3L, 𝐛_3L = unpack_params(𝚯_3L);
𝐖_3L::Vector{Matrix{Float32}}
𝐛_3L::Vector{Vector{Float32}}
𝐀_3L::Matrix{Float32} = 𝐖_3L[end];

𝐖_5L, 𝐛_5L = unpack_params(𝚯_5L);
𝐖_5L::Vector{Matrix{Float32}}
𝐛_5L::Vector{Vector{Float32}}
𝐀_5L::Matrix{Float32} = 𝐖_5L[end];

𝐖_7L, 𝐛_7L = unpack_params(𝚯_7L);
𝐖_7L::Vector{Matrix{Float32}}
𝐛_7L::Vector{Vector{Float32}}
𝐀_7L::Matrix{Float32} = 𝐖_7L[end];

𝐖_10L, 𝐛_10L = unpack_params(𝚯_10L);
𝐖_10L::Vector{Matrix{Float32}}
𝐛_10L::Vector{Vector{Float32}}
𝐀_10L::Matrix{Float32} = 𝐖_10L[end];

𝐖_20L, 𝐛_20L = unpack_params(𝚯_20L);
𝐖_20L::Vector{Matrix{Float32}}
𝐛_20L::Vector{Vector{Float32}}
𝐀_20L::Matrix{Float32} = 𝐖_20L[end];

#=
Defining Composition of Layers and Pushforwards/Jacobians:
=#

# 1-Layer
𝒩ₙₙ1L(x::Vector; ps=𝚯_1L::ComponentArray{<:Float32}, st=layer_states_1L) = model_1L(x, ps, st)[1]
# F₁1L(v)::Vector = ϕ.(W_01L * v + b_01L)
F₁1L(v)::Vector = ϕ.(𝐖_1L[1] * v + 𝐛_1L[1])
∂𝐳F₁ₗ(𝛍)::Matrix = Zygote.jacobian(𝛍 -> F₁1L(𝛍), 𝛍)[1];

# 2-Layers
𝐅_2L = Vector{Function}(undef, 2);
𝐅_2L::Vector{Function}
hiddenlayers_2L::Vector{Int64} = collect(keys(𝐖_2L))[1:end-1];
for i in hiddenlayers_2L
  𝐅_2L[i] = v -> ϕ.(𝐖_2L[i] * v + 𝐛_2L[i]);
end
𝒩ₙₙ2L(x::Vector; ps=𝚯_2L::ComponentArray{<:Float32}, st=layer_states_2L) = model_2L(x, ps, st)[1];
F₂ₗ(v::Vector)::Vector = reduce(∘, reverse(𝐅_2L))(v); # need reverse to get function composition order correct
∂𝐳F₂ₗ(μ::Vector)::Matrix = Zygote.jacobian(μ -> F₂ₗ(μ), μ)[1];

# 3-Layers
𝐅_3L = Vector{Function}(undef, 3);
𝐅_3L::Vector{Function};
hiddenlayers_3L::Vector{Int64} = collect(keys(𝐖_3L))[1:end-1];
for i in hiddenlayers_3L
  𝐅_3L[i] = v -> ϕ.(𝐖_3L[i] * v + 𝐛_3L[i]);
end
𝒩ₙₙ3L(x::Vector; ps=𝚯_3L::ComponentArray{<:Float32}, st=layer_states_3L) = model_3L(x, ps, st)[1];
F₃ₗ(v::Vector)::Vector = reduce(∘, reverse(𝐅_3L))(v::Vector);
∂𝐳F₃ₗ(μ::Vector)::Matrix = Zygote.jacobian(μ -> F₃ₗ(μ), μ)[1];

# 5-Layers
𝐅_5L = Vector{Function}(undef, 5);
𝐅_5L::Vector{Function};
hiddenlayers_5L::Vector{Int64} = collect(keys(𝐖_5L))[1:end-1];
for i in hiddenlayers_5L
  𝐅_5L[i] = v -> ϕ.(𝐖_5L[i] * v + 𝐛_5L[i]);
end

𝒩ₙₙ5L(x::Vector; ps=𝚯_5L::ComponentArray{<:Float32}, st=layer_states_5L) = model_5L(x, ps, st)[1];
F₅ₗ(v::Vector)::Vector = reduce(∘, reverse(𝐅_5L))(v);
∂𝐳F₅ₗ(μ::Vector)::Matrix = Zygote.jacobian(μ -> F₅ₗ(μ), μ)[1];

# 7-Layers
𝐅_7L = Vector{Function}(undef, 7);
𝐅_7L::Vector{Function};
hiddenlayers_7L::Vector{Int64} = collect(keys(𝐖_7L))[1:end-1];
for i in hiddenlayers_7L
  𝐅_7L[i] = v -> ϕ.(𝐖_7L[i] * v + 𝐛_7L[i]);
end

𝒩ₙₙ7L(x::Vector; ps=𝚯_7L::ComponentArray{<:Float32}, st=layer_states_7L) = model_7L(x, ps, st)[1];
F₇ₗ(v::Vector)::Vector = reduce(∘, reverse(𝐅_7L))(v);
∂𝐳F₇ₗ(μ::Vector)::Matrix = Zygote.jacobian(μ -> F₇ₗ(μ), μ)[1];

# 10-Layers
𝐅_10L = Vector{Function}(undef, 10);
𝐅_10L::Vector{Function};
𝐅_10L = 𝐅_10L;# |> Origin(0);
hiddenlayers_10L::Vector{Int64} = collect(keys(𝐖_10L))[1:end-1];
for i in hiddenlayers_10L
  𝐅_10L[i] = v -> ϕ.(𝐖_10L[i] * v + 𝐛_10L[i]);
end
𝒩ₙₙ10L(x::Vector; ps=𝚯_10L::ComponentArray{<:Float32}, st=layer_states_10L) = model_10L(x, ps, st)[1];
F₁₀ₗ(v::Vector)::Vector = reduce(∘, reverse(𝐅_10L))(v);
∂𝐳F₁₀ₗ(μ::Vector)::Matrix = Zygote.jacobian(μ -> F₁₀ₗ(μ), μ)[1];

# 12-Layers
# 𝐅_12L = Vector{Function}(undef, 12);
# 𝐅_12L::Vector{Function};
# hiddenlayers_12L::Vector{Int64} = collect(keys(𝐖_12L))[1:end-1];
# for i in hiddenlayers_12L
#   𝐅_12L[i] = v -> ϕ.(𝐖_12L[i] * v + 𝐛_12L[i])
# end
# 𝒩ₙₙ12L(x::Vector; ps=𝚯_12L::ComponentArray{<:Float32}, st=layer_states_12L) = model_12L(x, ps, st)[1];
# F₁₂ₗ(v::Vector)::Vector = reduce(∘, reverse(𝐅_12L))(v);
# ∂𝐳F₁₂ₗ(μ::Vector)::Matrix = Zygote.jacobian(μ -> F₁₂ₗ(μ), μ)[1];

# 18-Layers
# 𝐅_18L = Vector{Function}(undef, 18);
# 𝐅_18L::Vector{Function};
# 𝐅_18L = 𝐅_18L;# |> Origin(0);
# hiddenlayers_18L::Vector{Int64} = collect(keys(𝐖_18L))[1:end-1];
# for i in hiddenlayers_18L
#   𝐅_18L[i] = v -> ϕ.(𝐖_18L[i] * v + 𝐛_18L[i]);
# end
# 𝒩ₙₙ18L(x::Vector; ps=𝚯_18L::ComponentArray{<:Float32}, st=layer_states_18L) = model_18L(x, ps, st)[1];
# F₁₈ₗ(v::Vector)::Vector = reduce(∘, reverse(𝐅_18L))(v);
# ∂𝐳F₁₈ₗ(μ::Vector)::Matrix = Zygote.jacobian(μ -> F₁₈ₗ(μ), μ)[1];

# 20-Layers
𝐅_20L = Vector{Function}(undef, 20);
𝐅_20L::Vector{Function};
𝐅_20L = 𝐅_20L;# |> Origin(0);
hiddenlayers_20L::Vector{Int64} = collect(keys(𝐖_20L))[1:end-1];
for i in hiddenlayers_20L
  𝐅_20L[i] = v -> ϕ.(𝐖_20L[i] * v + 𝐛_20L[i])
end
𝒩ₙₙ20L(x::Vector; ps=𝚯_20L::ComponentArray{<:Float32}, st=layer_states_20L) = model_20L(x, ps, st)[1]
F₂₀ₗ(v::Vector)::Vector = reduce(∘, reverse(𝐅_20L))(v)
∂𝐳F₂₀ₗ(μ::Vector)::Matrix = Zygote.jacobian(μ -> F₂₀ₗ(μ), μ)[1]