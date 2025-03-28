import Lux: Chain, Dense, setup
import Random: Xoshiro

function create_1layer_model(Nx::Int, Ny::Int, σ, rng=Xoshiro(42))#, dev=cpu_device())

  # Our Pseudo-Random Number Generator
  # rng = Xoshiro(42)

  # Define the neural network model
  model::Chain = Chain(
      # Dense(Nx+1, 64, leakyrelu),
      Dense(Nx+1=>64, σ),
      # Dense(64, 64, σ),
      Dense(64=>Ny+1, identity; use_bias=false)
  )

  # ps, st
  parameters, operator_layer_states = setup(rng, model);

  return model, parameters, operator_layer_states
end


function create_2layer_model(Nx::Int, Ny::Int, σ::Activ_fns, rng=Xoshiro(42)) where {Activ_fns<:Function}

  # Our Pseudo-Random Number Generator
  # rng = Xoshiro(42)

  # Define the neural network model
  model::Chain = Chain( 
      # Dense(Nx+1, 64, leakyrelu),
      Dense(Nx+1, 64, σ),
      Dense(64, 64, σ),
      Dense(64, Ny+1, identity; use_bias=false)
  )

  # ps, st
  parameters, operator_layer_states = setup(rng, model);

  return model, parameters, operator_layer_states
end

function create_3layer_model(Nx::Int, Ny::Int, σ::Activ_fns, rng=Xoshiro(42)) where {Activ_fns<:Function}

  # Our Pseudo-Random Number Generator
  # rng = Xoshiro(42)

  # Define the neural network model
  model::Chain = Chain( 
      # Dense(Nx+1, 64, leakyrelu),
      Dense(Nx+1, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, Ny+1, identity; use_bias=false)
  )

  # ps, st
  parameters, operator_layer_states = setup(rng, model);

  return model, parameters, operator_layer_states
end

function create_4layer_model(Nx::Int, Ny::Int, σ::Activ_fns, rng=Xoshiro(42)) where {Activ_fns<:Function}

  # Our Pseudo-Random Number Generator
  # rng = Xoshiro(42)

  # Define the neural network model
  model::Chain = Chain(
      # Dense(Nx+1, 64, leakyrelu),
      Dense(Nx+1, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, Ny+1, identity; use_bias=false)
  )

  # ps, st
  parameters, operator_layer_states = setup(rng, model);

  return model, parameters, operator_layer_states
end

function create_5layer_model(Nx::Int, Ny::Int, σ::Activ_fns, rng=Xoshiro(42)) where {Activ_fns<:Function}

  # Our Pseudo-Random Number Generator
  # rng = Xoshiro(42)

  # Define the neural network model
  model::Chain = Chain(
      # Dense(Nx+1, 64, leakyrelu),
      Dense(Nx+1, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, Ny+1, identity; use_bias=false)
  )

  # ps, st
  parameters, operator_layer_states = setup(rng, model);

  return model, parameters, operator_layer_states
end

function create_10layer_model(Nx::Int, Ny::Int, σ::Activ_fns, rng=Xoshiro(42)) where {Activ_fns<:Function}

  # Our Pseudo-Random Number Generator
  # rng = Xoshiro(42)

  # Define the neural network model
  model::Chain = Chain(
      # Dense(Nx+1, 64, leakyrelu),
      Dense(Nx+1, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, 64, σ),
      Dense(64, Ny+1, identity; use_bias=false)
  )

  # ps, st
  parameters, operator_layer_states = setup(rng, model);

  return model, parameters, operator_layer_states
end

function create_nlayer_mlp(Nx::Int,
                            Ny::Int,
                            neuron_sizes::Vector{Int64},
                            𝛔::Vector{Activ_fns},
                            rng=Xoshiro(42);
                            num_hidden_layers::Int=1) where {Activ_fns<:Function}
  # Create a list to hold the layers
  layers = [];

  # Add the input layer
  # push!(layers, Dense(input_size+1, hidden_size, σ));
  push!(layers, Dense(Nx+1 => neuron_sizes[1], 𝛔[1]));

  # Add hidden layers
  # NOTE: num_layers is the number of maps between neurons = 2*
  for i in 2:(num_hidden_layers)
      # push!(layers, Dense(hidden_size, hidden_size, σ));
      push!(layers, Dense(neuron_sizes[i] => neuron_sizes[i],𝛔[i]))
  end

  # Add the output layer
  push!(layers, Dense(neuron_sizes[end] => Ny+1, identity; use_bias=false));

  # Construct the Lux chain
  mlp_model = Chain(layers...)

  parameters, operator_layer_states = setup(rng, mlp_model);
  return mlp_model, parameters, operator_layer_states
end
