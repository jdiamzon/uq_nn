### Linear Operator Files

#=--Test Local Operator--=#
#---1 Layer:
cuda_1layer_localLinop_file::String = "test_op/LinOP_1layerFF_leakyreluActFN_1MSamples_1000Epochs_1e-5LearnRate_1e-2RegRate"
cuda_1layer_localNonLinop_file::String = "test_op/NonLinOP_1layerFF_leakyreluActFN_1MSamples_1000Epochs_1e-5LearnRate_1e-2RegRate"
v2cuda_1layer_localLinop_file::String = "test_op/V2LinOP_1layerFF_leakyreluActFN_1MSamples_1000Epochs_1e-3LearnRate_1e-1RegRate"
v2cuda_1layer_localNonLinop_file::String = "test_op/V2_NonLinOP_1layerFF_leakyreluActFN_1MSamples_1000Epochs_1e-3LearnRate_1e-1RegRate"
NoRegcuda_1layer_localLinop_file::String = "test_op/NoReg/V2_LinOP_1layerFF_leakyreluActFN_1MSamples_1000Epochs_1e-3LearnRate_0RegRate"
#---2 Layers:
cuda_2layer_localLinop_file::String = "test_op/LinOP_1layerFF_leakyreluActFN_1MSamples_1000Epochs_1e-5LearnRate_1e-2RegRate"
cuda_2layer_localNonLinop_file::String = "test_op/NonLinOP_1layerFF_leakyreluActFN_1MSamples_1000Epochs_1e-5LearnRate_1e-2RegRate"
#---3 Layers:
cuda_3layer_localLinop_file::String = "test_op/3layerFF_leakyreluActFN_1MSamples_1kBatch_2000Epochs_1e-3LearnRate"
cuda_3layer_localNonLinop_file::String = "test_op/NonLinOP_1layerFF_leakyreluActFN_1MSamples_1000Epochs_1e-5LearnRate_1e-2RegRate"
#---5 Layers:
cuda_5layer_localLinop_file::String = "test_op/LinOP_1layerFF_leakyreluActFN_1MSamples_1000Epochs_1e-5LearnRate_1e-2RegRate"
cuda_5layer_localNonLinop_file::String = "test_op/NonLinOP_1layerFF_leakyreluActFN_1MSamples_1000Epochs_1e-5LearnRate_1e-2RegRate"
#=========================#

cpu_1layer_linop_file::String = "lin_op/CPU/1layerFF_100kSamples_1kEpochs"
# cuda_1layer_linop_file::String = "lin_op/CUDA/1layerFF_leakyreluActFN_1MSamples_1kBatch_2000Epochs_1e-3LearnRate"
cuda_1layer_linop_file::String = "lin_op/CUDA/1layerFF_leakyreluActFN_1MSamples_1kBatch_5000Epochs_dot001LR"
metal_1layer_linop_file::String = "lin_op/Metal/V3S1layerFF_leakyreluActFN_1MSamples_1kBatch_100Epochs_1e-4LearnRate"

cpu_2layer_linop_file::String = "lin_op/CPU/2layerFF_100kSamples_1kEpochs"
cuda_2layer_linop_file::String = "nonlin_op"
metal_2layer_linop_file::String = "lin_op/Metal/2layerFF_leakyreluActFN_1MSamples_1kBatch_100Epochs_1e-3LR"

# DO CPU
cpu_3layer_linop_file::String = "lin_op/CPU/3layerFF_1MSamples_1kBatch_100Epochs"
cuda_3layer_linop_file::String = "nonlin_op"

# DO CPU
cpu_5layer_linop_file::String = "lin_op/CPU/5layerFF_1MSamples_1kBatch_100Epochs"
cuda_5layer_linop_file::String = "nonlin_op"

cpu_7layer_linop_file::String = "lin_op/CPU/5layerFF_1MSamples_1kBatch_100Epochs"
cuda_7layer_linop_file::String = "lin_op/CUDA/7layerFF_leakyreluActFN_1MSamples_1kBatch_1000Epochs_dot001LR"

# cpu_10layer_linop_file::String = "lin_op/CPU/10layerFF_1MSamples_1kBatch_100Epochs"
# cuda_10layer_linop_file::String = "nonlin_op"

cpu_10layer_linop_file::String = "lin_op/CPU/10layerFF_1MSamples_1kBatch_100Epochs"
cuda_10layer_linop_file::String = "lin_op/CUDA/10layerFF_leakyreluActFN_1MSamples_1kBatch_1000Epochs_dot001LR"

cpu_12layer_linop_file::String = "lin_op/CPU/10layerFF_1MSamples_1kBatch_100Epochs"
cuda_12layer_linop_file::String = "lin_op/CUDA/12layerFF_leakyreluActFN_1MSamples_1kBatch_1000Epochs_dot1LR"

cpu_18layer_linop_file::String = "lin_op/CPU/18layerFF_1MSamples_1kBatch_100Epochs"
cuda_18layer_linop_file::String = "lin_op/CUDA/18layerFF_leakyreluActFN_1MSamples_1kBatch_1000Epochs_dot001LR"

#=
NOTE!!:  The learning rate for 20 layers has changed, due to convergence issues.
=#
cpu_20layer_linop_file::String = "lin_op/CPU/20layerFF_1MSamples_1kBatch_200Epochs"
cuda_20layer_linop_file::String = "lin_op/CUDA/20layerFF_leakyreluActFN_1MSamples_3.0e+03Batch_9.0e+02Epochs_1.0e-04LearnRate_1.0e+00RegRate_StepScheduler(true)WithDecayRate8.0e-01.jld2"

### Nonlinear Operator Files

#=--Test Local Operator--=#
cuda_1layer_localNonLinop_file::String = "test_op/NonLinOP_1layerFF_leakyreluActFN_1MSamples_1000Epochs_1e-5LearnRate_1e-2RegRate"
#=========================#

cpu_1layer_nonlinop_file::String = "lin_op/CPU/1layerFF_100kSamples_1kEpochs"
# cuda_1layer_nonlinop_file::String = "nonlin_op/CUDA/1layerFF_leakyreluActFN_1MSamples_1kBatch_2000Epochs_1e-3LearnRate"
cuda_1layer_nonlinop_file::String = "nonlin_op/CUDA/1layerFF_leakyreluActFN_1MSamples_2.0e+03Batch_1.0e+03Epochs_1.0e-02LearnRate_0.0e+00RegRate_StepScheduler(true)WithDecayRate8.0e-01"
cuda_1layer_nonlinop2_file::String = "nonlin_op2/CUDA/1layerFF_leakyreluActFN_1MSamples_1kBatch_2000Epochs_1e-3LearnRate"

cpu_2layer_nonlinop_file::String = "nonlin_op/CPU/2layerFF_LReluActFN_1MSamples_1kBatch_5000Epochs"
cuda_2layer_nonlinop_file::String = "nonlin_op/CUDA/2layerFF_leakyreluActFN_1MSamples_2.0e+03Batch_1.0e+03Epochs_1.0e-02LearnRate_0.0e+00RegRate_StepScheduler(true)WithDecayRate8.0e-01"
metal_2layer_nonlinop_file::String = "nonlin_op/Metal/2layerFF_leakyreluActFN_1MSamples_1kBatch_2000Epochs_1e-3LearnRat_1e-3RegRate"
cuda_2layer_nonlinop2_file::String = "nonlin_op2/CUDA/2layerFF_leakyreluActFN_1MSamples_1kBatch_5000Epochs_dot001LR"

cpu_3layer_nonlinop_file::String = "nonlin_op/CPU/3layerFF_LReluActFN_1MSamples_1kBatch_1000Epochs"
cuda_3layer_nonlinop_file::String = "nonlin_op/CUDA/3layerFF_leakyreluActFN_1MSamples_2.0e+03Batch_1.0e+03Epochs_1.0e-02LearnRate_0.0e+00RegRate_StepScheduler(true)WithDecayRate8.0e-01"
metal_3layer_nonlinop_file::String = "nonlin_op/Metal/3layerFF_leakyreluActFN_1MSamples_1kBatch_2000Epochs_1e-2LearnRate"
cuda_3layer_nonlinop2_file::String = "nonlin_op2/CUDA/3layerFF_leakyreluActFN_1MSamples_1kBatch_5000Epochs_dot001LR"

cpu_5layer_nonlinop_file::String = "nonlin_op/CPU/5layerFF_LReluActFN_1MSamples_1kBatch_1000Epochs"
cuda_5layer_nonlinop_file::String = "nonlin_op/CUDA/5layerFF_leakyreluActFN_1MSamples_2000.0Batch_300.0Epochs_0.01LearnRate_0RegRate_StepSchedulertruewithDecayRate0.8"
metal_5layer_nonlinop_file::String = "nonlin_op/Metal/V2_5layerFF_leakyreluActFN_1MSamples_1kBatch_2000Epochs_1e-2LearnRate"

# cpu_7layer_nonlinop_file::String = "nonlin_op/CPU/5layerFF_LReluActFN_1MSamples_1kBatch_1000Epochs"
cuda_7layer_nonlinop_file::String = "nonlin_op/CUDA/5layerFF_leakyreluActFN_1MSamples_2000.0Batch_300.0Epochs_0.01LearnRate_0RegRate_StepSchedulertruewithDecayRate0.8"
# metal_7layer_nonlinop_file::String = "nonlin_op/Metal/V2_5layerFF_leakyreluActFN_1MSamples_1kBatch_2000Epochs_1e-2LearnRate"

cpu_10layer_nonlinop_file::String = "nonlin_op/CPU/10layerFF_LReluActFN_1MSamples_1kBatch_1000Epochs"
cuda_10layer_nonlinop_file::String = "nonlin_op/CUDA/10layerFF_leakyreluActFN_1MSamples_2.0e+03Batch_3.0e+02Epochs_1.0e-02LearnRate_0.0e+00RegRate_StepScheduler(true)WithDecayRate8.0e-01"

cpu_12layer_nonlinop_file::String = "nonlin_op/CPU/12layerFF_LReluActFN_1MSamples_1kBatch_1000Epochs"
cuda_12layer_nonlinop_file::String = "nonlin_op/CUDA/12layerFF_leakyreluActFN_1MSamples_1kBatch_5000Epochs_dot001LR"

cpu_18layer_nonlinop_file::String = "lin_op/CPU/18layerFF_1MSamples_1kBatch_100Epochs"
cuda_18layer_nonlinop_file::String = "lin_op/CUDA/18layerFF_leakyreluActFN_1MSamples_1kBatch_1000Epochs_dot001LR"

cpu_20layer_nonlinop_file::String = "nonlin_op/CPU/nonlin_op"
cuda_20layer_nonlinop_file::String = "nonlin_op/CUDA/20layerFF_leakyreluActFN_1MSamples_3.0e+03Batch_9.0e+02Epochs_1.0e-04LearnRate_1.0e+00RegRate_StepScheduler(true)WithDecayRate8.0e-01"

### Dictionaries for Readability:

cpu_1layer::Dict{<:String} = Dict("Lin" => cpu_1layer_linop_file,
  "Nonlin" => cpu_1layer_linop_file)
cuda_1layer::Dict{<:String} = Dict("Lin" => cuda_1layer_linop_file,
  "Nonlin" => cuda_1layer_nonlinop_file,
  "Nonlin2" => cuda_1layer_nonlinop2_file,
  "TestLin" => cuda_1layer_localLinop_file,
  "TestNonLin" => cuda_1layer_localNonLinop_file,
  "TestLinV2" => v2cuda_1layer_localLinop_file,
  "TestNonLinV2" => v2cuda_1layer_localNonLinop_file,
  "TestLinNOREG" => NoRegcuda_1layer_localLinop_file)
metal_1layer::Dict{<:String} = Dict("Lin" => metal_1layer_linop_file)
file_1layer::Dict = Dict("CPU" => cpu_1layer, "CUDA" => cuda_1layer, "Metal" => metal_1layer)

cpu_2layer::Dict{<:String} = Dict("Lin" => cpu_2layer_linop_file,
  "Nonlin" => cpu_2layer_nonlinop_file)
cuda_2layer::Dict{<:String} = Dict("Lin" => cuda_2layer_linop_file,
  "Nonlin" => cuda_2layer_nonlinop_file,
  "Nonlin2" => cuda_2layer_nonlinop2_file)
metal_2layer::Dict{<:String} = Dict("Lin" => metal_2layer_linop_file,
  "Nonlin" => metal_2layer_nonlinop_file)
file_2layer::Dict = Dict("CPU" => cpu_2layer, "CUDA" => cuda_2layer, "Metal" => metal_2layer)

cpu_3layer::Dict{<:String} = Dict("Lin" => cpu_3layer_linop_file,
  "Nonlin" => cpu_3layer_nonlinop_file)
cuda_3layer::Dict{<:String} = Dict("Lin" => cuda_3layer_linop_file,
  "Nonlin" => cuda_3layer_nonlinop_file,
  "Nonlin2" => cuda_3layer_nonlinop2_file,)
metal_3layer::Dict{<:String} = Dict(
  "Nonlin" => metal_3layer_nonlinop_file)
file_3layer::Dict = Dict("CPU" => cpu_3layer, "CUDA" => cuda_3layer, "Metal" => metal_3layer)

cpu_5layer::Dict{<:String} = Dict("Lin" => cpu_5layer_linop_file,
  "Nonlin" => cpu_5layer_nonlinop_file)
cuda_5layer::Dict{<:String} = Dict("Lin" => cuda_5layer_linop_file,
  "Nonlin" => cuda_5layer_nonlinop_file)
metal_5layer::Dict{<:String} = Dict(
  "Nonlin" => metal_5layer_nonlinop_file)
file_5layer::Dict = Dict("CPU" => cpu_5layer, "CUDA" => cuda_5layer, "Metal" => metal_5layer)

cpu_7layer::Dict{<:String} = Dict("Lin" => cpu_7layer_linop_file,
  "Nonlin" => cuda_5layer_nonlinop_file)
cuda_7layer::Dict{<:String} = Dict("Lin" => cuda_7layer_linop_file,
  "Nonlin" => cuda_5layer_nonlinop_file)
file_7layer::Dict = Dict("CPU" => cpu_7layer, "CUDA" => cuda_7layer)

cpu_10layer::Dict{<:String} = Dict("Lin" => cpu_10layer_linop_file,
  "Nonlin" => cpu_10layer_nonlinop_file)
cuda_10layer::Dict{<:String} = Dict("Lin" => cuda_10layer_linop_file,
  "Nonlin" => cuda_10layer_nonlinop_file)
file_10layer::Dict = Dict("CPU" => cpu_10layer, "CUDA" => cuda_10layer)

cpu_12layer::Dict{<:String} = Dict("Lin" => cpu_12layer_linop_file,
  "Nonlin" => cuda_12layer_nonlinop_file)
cuda_12layer::Dict{<:String} = Dict("Lin" => cuda_12layer_linop_file,
  "Nonlin" => cuda_12layer_nonlinop_file)
file_12layer::Dict = Dict("CPU" => cpu_12layer, "CUDA" => cuda_12layer)

cpu_18layer::Dict{<:String} = Dict("Lin" => cpu_18layer_linop_file,
  "Nonlin" => cuda_18layer_nonlinop_file)
cuda_18layer::Dict{<:String} = Dict("Lin" => cuda_18layer_linop_file,
  "Nonlin" => cuda_18layer_nonlinop_file)
file_18layer::Dict = Dict("CPU" => cpu_18layer, "CUDA" => cuda_18layer)

cpu_20layer::Dict{<:String} = Dict(
  "Lin" => cpu_2layer_linop_file,
  "Nonlin" => cpu_2layer_nonlinop_file)
cuda_20layer::Dict{<:String} = Dict(
  "Lin" => cuda_20layer_linop_file,
  "Nonlin" => cuda_20layer_nonlinop_file)

file_20layer::Dict = Dict("CPU" => cpu_20layer, "CUDA" => cuda_20layer)