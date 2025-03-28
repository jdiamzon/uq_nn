import ComponentArrays: ComponentVector

#=
# Function to load parameters into their appropriate matrices
=#

function unpack_params(𝚯_Nlayers::T) where {T<:ComponentVector}

  𝐖_NL::Vector{Matrix{Float32}} = []
  𝐛_NL::Vector{Vector{Float32}} = []
  for layer_i in keys(𝚯_Nlayers)
    # if layer_i == keys(𝚯_Nlayers)[end]
    #   break
    # end

    push!(𝐖_NL, 𝚯_Nlayers[layer_i].weight)
    if haskey(𝚯_Nlayers[layer_i], :bias)
      push!(𝐛_NL, 𝚯_Nlayers[layer_i].bias)
    end
  end

  return 𝐖_NL, 𝐛_NL
end