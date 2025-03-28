include("../../Scripts/create_model.jl")
include("../../Scripts/data_generator.jl")
include("../../Scripts/Discretizations/gll_discretize.jl")
include("../../Scripts/sample_operators.jl")
include("../../Scripts/plot_data.jl")
# include("../../../custom_theming.jl")
using .GllDiscretize: Discretization, generate_gll_disc


using ADTypes,
  ComponentArrays,
  Functors,
  JLD2,
  KernelDensity,
  LinearAlgebra,
  Lux,
  # LuxCUDA,  # Removed since we are training on CPU
  MLUtils,
  Random,
  Zygote,
  Optimisers,
  Printf,
  Distributions

# Initialize a random number generator (no device-specific seeding is required for CPU)
rng = MersenneTwister(12345)
Random.seed!(rng, 12345)

#----------------------------------------------------------------------------
# Model Training Functions (CPU version)
#----------------------------------------------------------------------------

function rand_func(x)
  return randn(length(x))
end

function train_model_with_dataloader(model, ps, st, dataloader; epochs, lr)
  train_loss_history = Float64[]
  train_state = Training.TrainState(model, ps, st, Adam(lr))

  for epoch in 1:epochs
    epoch_loss = 0.0
    num_batches = 0

    for (i, (x_batch, y_batch)) in enumerate(dataloader)
      # For CPU training, you no longer need to transfer batches to the GPU.
      # (i.e. we remove any 'cu' conversion calls)

      # Perform a training step using AutoZygote and the MSE loss
      _, loss, _, train_state = Training.single_train_step!(AutoZygote(), MSELoss(), (x_batch, y_batch), train_state)

      epoch_loss += loss
      num_batches += 1

      if epoch % 10 == 0 && i % 100 == 0
        println("Epoch: [$epoch/$epochs]\tBatch: $i\tLoss: $loss")
      end
    end

    avg_epoch_loss = epoch_loss / num_batches
    push!(train_loss_history, avg_epoch_loss)

    if epoch % 10 == 0
      println("Epoch: [$epoch/$epochs]\tAverage Loss: $avg_epoch_loss")
    end
  end

  return train_state, train_loss_history
end

function loss_function(model, ps, st, x, y)
  pred, _ = model(x, ps, st)
  return MSELoss()(pred, y)
end

#----------------------------------------------------------------------------
# Generate Training Data and Train (CPU version)
#----------------------------------------------------------------------------

function train_model_with_params(ϕ::Function, ℒ_op::Function;
    NN_width::Int,
    NN_depth::Int,
    Nx::Int = 30,
    Ny::Int = 30,
    num_samples::Int = 1_000_000,
    selected_epochs::Int = 5000,
    learn_rate = 0.001)

  # Generate data
  _, _, train_input_fns, train_output_fns = generate_data(Nx, Ny, rand_func, ℒ_op, num_samples)
  train_input_fns = f32(train_input_fns)
  train_output_fns = f32(train_output_fns)

  # Define network architecture
  neuron_sizes = fill(NN_width, NN_depth + 1)
  𝛟 = fill(ϕ, NN_depth + 1)

  # Create the model (chain) and its parameters/states
  model, Θ, layer_states = create_nlayer_mlp(Nx, Ny, neuron_sizes, 𝛟; num_hidden_layers = NN_depth)

  # For CPU training, we keep the parameters and state as is (no device transfers)
  # Θ_cpu = Θ (or simply use Θ), same for layer_states

  # Set up the DataLoader for batch learning
  dataloader = DataLoader((train_input_fns, train_output_fns); parallel = true, batchsize = 1000)

  # Train the model on CPU
  train_state, train_loss_history = train_model_with_dataloader(model, Θ, layer_states, dataloader;
                                                                epochs = selected_epochs, lr = learn_rate)

  return train_state, model, Θ, layer_states
end
