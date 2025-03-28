using Distributions: Distribution

function generate_noised_samples(𝛍_input::Vector;
  𝒩ₙₙ::Function,
  num_samples::Int=100_000,
  noise_pdf::Distribution)
  ### === Noise Generation
  input_noise_samples::Matrix = zeros(length(𝛍_input), num_samples);
  #=
    We're treating the output space as having same dim as input space.
  =#
  # output_noised_samples::Matrix = zeros(length(output_data), num_samples)
  output_noised_samples::Matrix = zeros(length(𝛍_input), num_samples)

  for n in 1:num_samples
    𝐳_noise::Vector = rand(noise_pdf, length(𝛍_input));
    input_noise_samples[:, n] = 𝐳_noise;
    output_noised_samples[:, n] = 𝒩ₙₙ(𝛍_input .+ input_noise_samples[:, n])
  end

  return input_noise_samples, output_noised_samples
end