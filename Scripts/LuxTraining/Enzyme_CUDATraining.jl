include("../../Scripts/create_model.jl")
include("../../Scripts/data_generator.jl")
include("../../Scripts/Discretizations/gll_discretize.jl")
include("../../Scripts/sample_operators.jl")
include("../../Scripts/plot_data.jl")
include("../../../custom_theming.jl")

using ADTypes,
  Enzyme,              # ← Use Enzyme for AD
  ComponentArrays,
  Functors,
  JLD2,
  KernelDensity,
  LinearAlgebra,
  Lux,
  # LuxCUDA,
  Metal,
  MLUtils,
  Random,
  Reactant,           # ← Use Reactant for device management
  Optimisers,
  Printf,
  Distributions,
  CairoMakie

# Set the theme and activate the plotting backend.
set_theme!(dankcula_theme)
CairoMakie.activate!()

# Define the loss (here we use mean-squared error)
const loss_function = MSELoss()

# Define our devices for parameters and data
const cdev = cpu_device()
const xdev = reactant_device()

# Initialize a random number generator
rng = MersenneTwister(12345)
Random.seed!(rng, 12345)

# A simple random function for generating input curves
function rand_func(x)
  return randn(length(x))
end

#-------------------------------------------------------------------------
# Training loop (using Enzyme for AD and Reactant for device transfers)
#-------------------------------------------------------------------------
function train_model_with_dataloader(model, Θ, st, dataloader; epochs, lr)
  train_loss_history = Float64[]
  # Set up the training state with an Adam optimizer.
  train_state = Training.TrainState(model, Θ, st, Optimisers.Adam(lr))
  
  for epoch in 1:epochs
    epoch_loss = 0.0
    num_batches = 0
    for (i, (x_batch, y_batch)) in enumerate(dataloader)
      # Transfer data to our target device via Reactant
      x_batch = xdev(x_batch)
      y_batch = xdev(y_batch)
      
      # Here we perform one training step. Note that in the updated Lux
      # tutorials, we pass the Enzyme AD backend instead of AutoZygote.
      _, loss, _, train_state = Training.single_train_step!(
          AutoEnzyme(),             # <-- use Enzyme here
          MSELoss(),          # <-- loss function (MSE)
          (x_batch, y_batch), 
          train_state)
      
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

# (Optional) A wrapper to compute the loss given the model and its parameters.
function model_loss(model, Θ, st, x, y)
  pred, _ = model(x, Θ, st)
  return MSELoss()(pred, y)
end

#-------------------------------------------------------------------------
# Generate training data and train the operator-learning model.
#-------------------------------------------------------------------------
function train_model_with_params(ϕ::Function, ℒ_op::Function; 
      NN_width::Int, NN_depth::Int, 
      Nx::Int=30, Ny::Int=30, 
      num_samples::Int=1_000_000, 
      selected_epochs::Int=5000, 
      selected_learn_rate=0.001,
      selected_batchsize=1000)

  # Generate the (input, output) training curves.
  _, _, train_input_fns, train_output_fns = generate_data(Nx, Ny, rand_func, ℒ_op, num_samples)
  train_input_fns = f32(train_input_fns)
  train_output_fns = f32(train_output_fns)
  
  # Build a multilayer perceptron (MLP) model.
  neuron_sizes = fill(NN_width, NN_depth + 1)
  𝛟 = fill(ϕ, NN_depth + 1)
  model, Θ, layer_states = create_nlayer_mlp(Nx, Ny, neuron_sizes, 𝛟; num_hidden_layers=NN_depth)
  
  # Transfer the parameters and layer states to the desired device.
  Θ_gpu = Θ |> xdev
  st_gpu = layer_states |> xdev
  
  # Create a data loader for batching.
  dataloader = DataLoader((train_input_fns, train_output_fns); parallel=true, batchsize=selected_batchsize)
  
  # Train the model using our updated training loop.
  train_state, train_loss_history = train_model_with_dataloader(model, Θ_gpu, st_gpu, dataloader; 
                                                                epochs=selected_epochs, lr=selected_learn_rate)
  # Transfer the final parameters back to the CPU.
  Θ_cpu = Θ_gpu |> cdev
  st_cpu = st_gpu |> cdev
  
  return train_state, model, Θ_cpu, st_cpu
end