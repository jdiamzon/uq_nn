Π(x; radius=1.0)::Vector = heaviside.(x .+ radius) - heaviside.(x .- radius)

##===##  Calculation of Approximated Moments

#=
function 


ω(gⱼ)::Vector = gⱼ .- dot(αⱼ5L, F₅ₗ(μ_part))
J::Matrix = ∂𝐳F₅ₗ(μ_part)
Q::Matrix = 𝐀_5L * J
q_j::Vector = Q[selection, :]
gₖ::Vector = output_noised_g_part_samples[selection, :]

γ::Vector = vec(transpose(αⱼ5L) * J)
ℓ::Vector = β * γ



ρ_out::Vector = 𝐀_5L * F₅ₗ(μ_part)

mean_from_formula(outputᵢ::Int)::Real = dot(𝐀_5L[outputᵢ, :], F₅ₗ(μ_part))
mean_prod(outputᵢ::Int, outputⱼ::Int)::Real = ρ_out[outputᵢ] * ρ_out[outputⱼ] + ((β^2) / 3.0) * dot(Q[outputᵢ, :], Q[outputⱼ, :])
var_from_formula(outputᵢ::Int)::Real = mean_prod(outputᵢ, outputᵢ) - mean_from_formula(outputᵢ)^2
Cov_formula(outputᵢ::Int, outputⱼ::Int)::Real = mean_prod(outputᵢ, outputⱼ) - mean_from_formula(outputᵢ) * mean_from_formula(outputⱼ)

##===## Calculation of Approximated Moments
=#

##===##  Product of Sinc Formula

sinc_func(x, γ) = sin(γ * x) / (γ * x)

#=
# test_sinc_func = sinc_func.(x, ℓ[1]);
# amount = Nx + 1
# for k in 2:(amount)
#   sinc_values_k = sinc_func.(x, ℓ[k])
#   test_sinc_func = test_sinc_func .* sinc_values_k
# end
# test_sinc_func = N * (1.0 / (2.0 * π)) * test_sinc_func

# test_test_sinc_func = zeros(Nx+1, length(x));
# for k in 1:(Nx + 1)
#   test_test_sinc_func[k,:] = sinc_func.(x, ℓ[k]);
# end

# test_sinc_func = zeros(Nx+1, length(x));
# for k in 1:(Nx + 1)
#   test_sinc_func[k,:] = sinc_func.(x, ℓ[k]);
# end
# test_sinc_func = N * (1.0 / (2.0 * π)) *  prod(test_sinc_func, dims=1)
=#

function Πsinc(ℓ::Vector, 𝐚::LinRange; Npoints::Int)
  # result_sinc = zeros(Nx + 1, length(𝐚))
  result_sinc = zeros(Npoints + 1, length(𝐚))
  for k in 1:(Npoints+1)
    result_sinc[k, :] = sinc_func.(𝐚, ℓ[k])
  end
  result_sinc = vec(prod(result_sinc, dims=1))
end

##===##  Product of Sinc Formula