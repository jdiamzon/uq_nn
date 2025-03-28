include("../../Scripts/utils/create_model.jl")
include("../../Scripts/data_generator.jl")
include("../../Scripts/Discretizations/gll_discretize.jl")
include("../../Scripts/sample_operators.jl")
# include("../../Scripts/plot_data.jl")
# include("../../../custom_theming.jl")
using .GllDiscretize: Discretization, generate_gll_disc

using ADTypes,
  # DataFrames,
  # CairoMakie, # Primary Plotting library
  # Enzyme,
  ComponentArrays,
  # FiniteDifferences,
  Functors,
  JLD2, # Used for saving
  KernelDensity,
  LinearAlgebra,
  Lux,
  LuxCUDA,
  MLUtils,
  Random,
  # Reactant,
  # RemoteREPL,
  Zygote,
  Optimisers,
  ParameterSchedulers,
  # Polynomials,
  Printf,
  Distributions
  # Statistics
# set_theme!(dankcula_theme)
# CairoMakie.activate!()

#=
Initializing a random number generator
=#
const dev_gpu::CUDADevice = gpu_device();
const dev_cpu::CPUDevice = cpu_device();
rng::Random.MersenneTwister = MersenneTwister();
Random.seed!(rng, 12345);

#=
Model Training Functions
=#

function rand_func(x)
  return randn(length(x))
end

function train_model_with_dataloader(model, ps, st, dataloader;
  epochs,
  lr)
  # dec_of_p,
  # rr)

  train_loss_history::Vector{Float64} = []
  train_state = Training.TrainState(model, ps, st, Optimisers.Adam(lr))

  for epoch in 1:epochs
    epoch_loss::Real = 0.0
    num_batches::Int = 0
    for (i, (x_batch, y_batch)) in enumerate(dataloader)
      # Transfer to the desired device
      x_batch = cu(x_batch)
      y_batch = cu(y_batch)

      # Perform a training step in one sixth_conv_test
      _, loss, _, train_state = Training.single_train_step!(
        AutoZygote(), MSELoss(), (x_batch, y_batch), train_state)

      # Accumulate loss and batch count
      epoch_loss += loss
      num_batches += 1

      # Print loss for monitoring
      # if epoch % 10 == 0 || i == 1
      if epoch % 10 == 0 && i % 100 == 0
        println("Epoch: [$epoch/$epochs]\tBatch: $i\tLoss: $loss")
      end
    end

    # Compute average loss for the epoch
    avg_epoch_loss::Float64 = epoch_loss / num_batches
    push!(train_loss_history, avg_epoch_loss)

    # Print average loss for the epoch
    if epoch % 10 == 0
      println("Epoch: [$epoch/$epochs]\tAverage Loss: $avg_epoch_loss")
    end

  end

  return train_state, train_loss_history
end

function train_model_with_dataloader_reg(model, ps, st, dataloader;
  epochs,
  lr,
  dec_of_p,
  rr,
  paramScheduling::Bool,
  dr)

  train_loss_history::Vector{Float64} = []
  train_state = Training.TrainState(model, ps, st, Optimisers.AdamW(lr,dec_of_p,rr))

  s = ParameterSchedulers.Stateful(Step(start = lr, decay = dr, step_sizes = collect(1:10:epochs)))

  for epoch in 1:epochs
    epoch_loss::Real = 0.0
    num_batches::Int = 0
    for (i, (x_batch, y_batch)) in enumerate(dataloader)
      # Transfer to the desired device
      x_batch = cu(x_batch)
      y_batch = cu(y_batch)

      # Perform a training step in one sixth_conv_test
      _, loss, _, train_state = Training.single_train_step!(
        AutoZygote(), MSELoss(), (x_batch, y_batch), train_state)

      # Accumulate loss and batch count
      epoch_loss += loss
      num_batches += 1

      # Print loss for monitoring
      # if epoch % 10 == 0 || i == 1
      if epoch % 10 == 0 && i % 100 == 0
        println("Epoch: [$epoch/$epochs]\tBatch: $i\tLoss: $loss")
      end
    end

    # Compute average loss for the epoch
    avg_epoch_loss::Float64 = epoch_loss / num_batches
    push!(train_loss_history, avg_epoch_loss)

    # Print average loss for the epoch
    if epoch % 10 == 0
      println("Epoch: [$epoch/$epochs]\tAverage Loss: $avg_epoch_loss")
    end

    if paramScheduling == true
      Optimisers.adjust!(train_state, ParameterSchedulers.next!(s))
    end

  end

  return train_state, train_loss_history
end

function loss_function(model, ps, st, x, y)
  pred, _ = model(x, ps, st)
  return MSELoss()(pred, y)
end

#=
Generate Training Data and Train
=#

function train_model_with_params_reg(ϕ::Function, ℒ_op::Function;
      NN_width::Int, NN_depth::Int, 
      Nx::Int=30, Ny::Int=30, 
      num_samples::Int=1_000_000, 
      selected_epochs::Int=5000, 
      selected_learn_rate=0.001,
      selected_reg_rate=0.0,
      selected_batchsize=1000,
      selected_decOfMomentums = (0.9, 0.999),
      doWeParamSchedule::Bool = false,
      selected_decay_rate=0.0)

_,_,train_input_fns::Matrix, train_output_fns::Matrix = generate_data(Nx, Ny, rand_func, ℒ_op, num_samples);

train_input_fns = f32(train_input_fns);
train_output_fns = f32(train_output_fns);

neuron_sizes::Vector{Int} = fill(NN_width, NN_depth + 1);
𝛟::Vector{<:Function} = fill(ϕ, NN_depth + 1);

model::Chain, Θ::NamedTuple, layer_states::NamedTuple = create_nlayer_mlp(Nx,
  Ny,
  neuron_sizes,
  𝛟; num_hidden_layers=NN_depth);

Θ_gpu::NamedTuple = Θ |> dev_gpu;
st_gpu::NamedTuple = layer_states |> dev_gpu;

train_loss_history::Vector{Float64} = [];
dataloader::MLUtils.DataLoader = DataLoader((train_input_fns, train_output_fns); parallel=true, batchsize=selected_batchsize)

train_state, train_loss_history = train_model_with_dataloader_reg(model, Θ_gpu, st_gpu, dataloader;
epochs=selected_epochs,
lr=selected_learn_rate,
dec_of_p=selected_decOfMomentums,
rr=selected_reg_rate,
paramScheduling=doWeParamSchedule,
dr=selected_decay_rate)

Θ_cpu = Θ_gpu |> dev_cpu
st_cpu = st_gpu |> dev_cpu

return train_state, train_loss_history, model, Θ_cpu, st_cpu

end

function train_model_with_params(ϕ::Function, ℒ_op::Function;
  NN_width::Int, NN_depth::Int, 
  Nx::Int=30, Ny::Int=30, 
  num_samples::Int=1_000_000, 
  selected_epochs::Int=5000, 
  selected_learn_rate=0.001,
  # selected_reg_rate=0.0,
  selected_batchsize=1000)
  # selected_decOfMomentums = (0.9, 0.999))

_,_,train_input_fns::Matrix, train_output_fns::Matrix = generate_data(Nx, Ny, rand_func, ℒ_op, num_samples);

train_input_fns = f32(train_input_fns);
train_output_fns = f32(train_output_fns);

neuron_sizes::Vector{Int} = fill(NN_width, NN_depth + 1);
𝛟::Vector{<:Function} = fill(ϕ, NN_depth + 1);

model::Chain, Θ::NamedTuple, layer_states::NamedTuple = create_nlayer_mlp(Nx,
Ny,
neuron_sizes,
𝛟; num_hidden_layers=NN_depth);

Θ_gpu::NamedTuple = Θ |> dev_gpu;
st_gpu::NamedTuple = layer_states |> dev_gpu;

train_loss_history::Vector{Float64} = [];
dataloader::MLUtils.DataLoader = DataLoader((train_input_fns, train_output_fns); parallel=true, batchsize=selected_batchsize)

train_state, train_loss_history = train_model_with_dataloader(model, Θ_gpu, st_gpu, dataloader;
epochs=selected_epochs,
lr=selected_learn_rate)
# dec_of_p=selected_decOfMomentums,
# rr=selected_reg_rate)

Θ_cpu = Θ_gpu |> dev_cpu
st_cpu = st_gpu |> dev_cpu

return train_state, model, Θ_cpu, st_cpu

end