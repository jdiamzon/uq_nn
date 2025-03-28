using CairoMakie, PairPlots
using ComponentArrays: ComponentVector
using DataFrames
using Distributions
using FourierTools: ift
using LinearAlgebra: dot
using LaTeXStrings
using Format
include("approx_formula.jl")
include("../noise_generator.jl")
include("unpack_params.jl")
interp_interval = collect(range(-1.0, 1.0, 100))

@kwdef struct Data_with_Grid{T}
  input_data::Vector{T}
  output_data::Vector{T}
  nodes::Vector{T}
end

function add_interp_line(ax::Makie.Axis, points_data::Vector, nodes_x)
  poly_fit = Polynomials.fit(nodes_x, points_data)
  Makie.lines!(ax, interp_interval, poly_fit.(interp_interval))
  # label=curve_label; linewidth=select_linewidth,alpha=select_alpha)
end

function add_interp_line(ax::Makie.Axis, points_data::Vector, select_linewidth::Float64, select_alpha::Float64; selected_color::Symbol)
  poly_fit = Polynomials.fit(nodes_x, points_data)
  Makie.lines!(ax, interp_interval, poly_fit.(interp_interval); linewidth=select_linewidth, alpha=select_alpha, color=selected_color)
  # label=curve_label; linewidth=select_linewidth,alpha=select_alpha)
end

function add_interp_line(ax::Makie.Axis, points_data::Vector, curve_label::LaTeXStrings.LaTeXString, select_linewidth::Float64, select_alpha::Float64; selected_color::Symbol)
  poly_fit = Polynomials.fit(nodes_x, points_data)
  Makie.lines!(ax, interp_interval, poly_fit.(interp_interval), label=curve_label; linewidth=select_linewidth, alpha=select_alpha, color=selected_color)
  # label=curve_label; linewidth=select_linewidth,alpha=select_alpha)
end


function add_interp_line(ax::Makie.Axis, points_data::Vector, curve_label::LaTeXStrings.LaTeXString, select_linewidth, selected_linestyle,selected_color::Symbol)
  poly_fit = Polynomials.fit(nodes_x, points_data)
  Makie.lines!(ax, interp_interval, poly_fit.(interp_interval), label=curve_label; linewidth=select_linewidth,linestyle=selected_linestyle,color=selected_color)
end


function plot_inputoutput_curves(fₚ_desc::Data_with_Grid, 𝒩ₙₙNL::Function, 𝚯_NL::Params; plot_size=(900, 500)) where {Params<:ComponentVector}

  input_data::Vector = fₚ_desc.input_data
  output_data::Vector = fₚ_desc.output_data
  grid_nodes::Vector = fₚ_desc.nodes


  num_nonHidden_layer::Int = 1
  num_layers::Int = length(keys(𝚯_NL)) - num_nonHidden_layer
  y_sol::Vector = output_data
  # y_pred::Vector = 𝒩ₙₙNL(input_data; ps=𝚯_NL, st=layer_states_NL)
  y_pred::Vector = 𝒩ₙₙNL(input_data)

  begin
    fig = Figure(size=plot_size)
    ax1 = Makie.Axis(fig[1, 1];
      xlabel="x",
      ylabel=L"y = f(x)",
      title=L"\text{(%$num_layers Layers) Input function: } f(x) = exp(sin(5x))")

    add_interp_line(ax1, input_data, grid_nodes)
    CairoMakie.scatter!(ax1, grid_nodes, input_data,
      label="\"Sampled\" Data";
      markersize=12,
      alpha=0.5,
      strokewidth=2)

    axislegend(ax1)
    #=
    Second figure below
    =#

    ax2 = Makie.Axis(fig[1, 2];
      xlabel="x",
      ylabel=L"y = \mathcal{N}(f(x))",
      title=L"\text{(%$num_layers Layers) Output function: } \mathcal{N}(f(x)) \text{ vs. NN Solution}")

    add_interp_line(ax2, output_data, grid_nodes)
    Makie.scatter!(ax2, grid_nodes, output_data,
      label="\"Sampled\" Data";
      markersize=12,
      alpha=0.5,
      strokewidth=2)
    Makie.scatter!(ax2, grid_nodes, y_pred,
      label="Neural Net Generated Data";
      markersize=6,
      alpha=0.5,
      strokewidth=2)

    axislegend(ax2)
    fig
  end
end

function plot_output_marginals(f_input::Vector;
  selection::Int=rand(1:length(f_input)),
  𝒩ₙₙNL::NeuralNet,
  𝚯_NL::Params,
  Fₙₗ::F,
  ∂𝐳Fₙₗ::D,
  β_noise::Real=1.5,
  noise_distrib::Distribution,
  chosen_pt_per_unit::Real = 0.5) where {D<:Function,
  F<:Function,
  NeuralNet<:Function,Params<:ComponentVector}

  _, output_noised_g_part_samples_NL = generate_noised_samples(f_input;
    𝒩ₙₙ=𝒩ₙₙNL,
    noise_pdf=noise_distrib)

  num_nonHidden_layers::Int = 1
  num_layers::Int = length(keys(𝚯_NL)) - num_nonHidden_layers

  #===
  Recall the parameters we have:
  ===#
  # e.g. selection = 14
  # selection::Int = rand(1:length(f_input))
  #=
  Recall: Nx = Ngll_grid - 1
  =#
  Nx::Int = length(f_input) - 1
  μₚ::Vector = f_input
  𝐀_NL::Matrix = unpack_params(𝚯_NL)[1][end]
  αᵢNL::Vector = 𝐀_NL[selection, :]

  ∂𝐳Fₙₗ_ofμₚ::Matrix = ∂𝐳Fₙₗ(μₚ)
  Q::Matrix = 𝐀_NL * ∂𝐳Fₙₗ_ofμₚ
  # q_i::Vector = Q[selection, :]
  gᵢ::Vector = output_noised_g_part_samples_NL[selection, :]

  γ::Vector = vec(transpose(αᵢNL) * ∂𝐳Fₙₗ_ofμₚ)
  ℓ::Vector = β_noise * γ

  ##=== Calculation of IFT
  N::Int = 2048
  L::Int = N / 2
  x::LinRange = LinRange(-L, L, N)

  𝒻::Vector = N * (1.0 / (2.0 * π)) * Πsinc(ℓ, x; Npoints=Nx)

  # Step 2: Perform the inverse Fourier transform
  𝒻̂ = ift(𝒻)
  ##=== Calculation of IFT

  ##=== Calculation of Approximated Moments

  ρ_out::Vector = 𝐀_NL * Fₙₗ(μₚ)

  ω_shift(xᵢ::LinRange; μ::Vector)::Vector = xᵢ .+ dot(αᵢNL, Fₙₗ(μ))
  mean_from_formula(outputᵢ::Int)::Real = dot(𝐀_NL[outputᵢ, :], Fₙₗ(μₚ))
  mean_prod(outputᵢ::Int, outputⱼ::Int)::Real = ρ_out[outputᵢ] * ρ_out[outputⱼ] + ((β_noise^2) / 3.0) * dot(Q[outputᵢ, :], Q[outputⱼ, :])
  var_from_formula(outputᵢ::Int)::Real = mean_prod(outputᵢ, outputᵢ) - mean_from_formula(outputᵢ)^2
  Cov_from_formula(outputᵢ::Int, outputⱼ::Int)::Real = mean_prod(outputᵢ, outputⱼ) - mean_from_formula(outputᵢ) * mean_from_formula(outputⱼ)

  𝔼gᵢ::Real = mean_from_formula(selection)
  𝕍gᵢ::Real = var_from_formula(selection)

  ##=== Calculation of Approximated Moments

  fig::Makie.Figure = Figure(size=(1000, 1000), pt_per_unit=chosen_pt_per_unit, fontsize=35)

  begin
    pdf_height = maximum(kde(gᵢ).density)
    ax = fig[1,1] = Makie.Axis(fig;
      limits=(mean(gᵢ) - 3.0 * std(gᵢ), mean(gᵢ) + 3.0 * std(gᵢ), 0, pdf_height + 0.2),
      xlabel=L"g_{%$selection} = \mathcal{N}(f(μ_{%$selection} \pm \beta))",
      ylabel=L"p(g_{%$selection})",
      title=L"p(g_{%$selection}):\text{%$num_layers Layers}, \quad |\beta| \leq %$β")
    hist!(ax, gᵢ, normalization=:pdf, bins=100, label_formatter=x -> round(x, digits=2), label_size=15, strokewidth=0.5, strokecolor=(:black, 0.5), color=:values)
    # Makie.density!(gᵢ, label="KDE Estimation")

    CairoMakie.lines!(ax, x * (π / L) .+ dot(αᵢNL, Fₙₗ(μₚ)), abs.(𝒻̂), label=L"\mathcal{F}^{-1}(\Pi_{k=1}^{%$(Nx+1)}sinc(\ell_k a_{%$selection}))", color=:darkgoldenrod, linewidth=3.5)
    mean_value = Format.format(mean(gᵢ), precision=4)
    mean_val_from_formula = Format.format(𝔼gᵢ, precision=4)
    var_value = Format.format(var(gᵢ), precision=4)
    var_val_from_formula = Format.format(𝕍gᵢ, precision=4)

    vlines!(ax, mean(gᵢ), label=L"\mathbb{E}[g_k]_{\text{num}} = %$mean_value", linestyle=:dash)
    vlines!(ax, 𝔼gᵢ, label=L"\mathbb{E}[g_k]_{\text{formula}} = %$mean_val_from_formula", linestyle=:dash)
    vlines!(ax, var(gᵢ), label=L"\mathbb{V}[g_k]_{\text{num}} = %$var_value", linestyle=:dash, linewidth=6.5,color=:transparent)
    vlines!(ax, 𝕍gᵢ, label=L"\mathbb{V}[g_k]_{\text{formula}} = %$var_val_from_formula", linestyle=:dash, linewidth=6.5,color=:transparent)

    # CURRENTLY BROKEN with Makie.density
    fig[1, 2] = Legend(fig, ax, framevisible = false)
  end
  return fig
end

function plot_output_marginals_noLegend(f_input::Vector;
  selection::Int=rand(1:length(f_input)),
  𝒩ₙₙNL::NeuralNet,
  𝚯_NL::Params,
  Fₙₗ::F,
  ∂𝐳Fₙₗ::D,
  β_noise::Real=1.5,
  noise_distrib::Distribution,
  chosen_offset::Real=0.5,
  chosen_pt_per_unit::Real=0.5) where {D<:Function,
  F<:Function,
  NeuralNet<:Function,Params<:ComponentVector}

  _, output_noised_g_part_samples_NL = generate_noised_samples(f_input;
    𝒩ₙₙ=𝒩ₙₙNL,
    noise_pdf=noise_distrib)

  num_nonHidden_layers::Int = 1
  num_layers::Int = length(keys(𝚯_NL)) - num_nonHidden_layers

  #===
  Recall the parameters we have:
  ===#
  # e.g. selection = 14
  # selection::Int = rand(1:length(f_input))
  #=
  Recall: Nx = Ngll_grid - 1
  =#
  Nx::Int = length(f_input) - 1
  μₚ::Vector = f_input
  𝐀_NL::Matrix = unpack_params(𝚯_NL)[1][end]
  αᵢNL::Vector = 𝐀_NL[selection, :]

  ∂𝐳Fₙₗ_ofμₚ::Matrix = ∂𝐳Fₙₗ(μₚ)
  Q::Matrix = 𝐀_NL * ∂𝐳Fₙₗ_ofμₚ
  gᵢ::Vector = output_noised_g_part_samples_NL[selection, :]

  γ::Vector = vec(transpose(αᵢNL) * ∂𝐳Fₙₗ_ofμₚ)
  ℓ::Vector = β_noise * γ

  ##=== Calculation of IFT
  N::Int = 2048
  L::Int = N / 2
  x::LinRange = LinRange(-L, L, N)

  𝒻::Vector = N * (1.0 / (2.0 * π)) * Πsinc(ℓ, x; Npoints=Nx)

  # Step 2: Perform the inverse Fourier transform
  𝒻̂ = ift(𝒻)
  ##=== Calculation of IFT

  ##=== Calculation of Approximated Moments

  ρ_out::Vector = 𝐀_NL * Fₙₗ(μₚ)

  ω_shift(xᵢ::LinRange; μ::Vector)::Vector = xᵢ .+ dot(αᵢNL, Fₙₗ(μ))
  mean_from_formula(outputᵢ::Int)::Real = dot(𝐀_NL[outputᵢ, :], Fₙₗ(μₚ))
  mean_prod(outputᵢ::Int, outputⱼ::Int)::Real = ρ_out[outputᵢ] * ρ_out[outputⱼ] + ((β_noise^2) / 3.0) * dot(Q[outputᵢ, :], Q[outputⱼ, :])
  var_from_formula(outputᵢ::Int)::Real = mean_prod(outputᵢ, outputᵢ) - mean_from_formula(outputᵢ)^2
  Cov_from_formula(outputᵢ::Int, outputⱼ::Int)::Real = mean_prod(outputᵢ, outputⱼ) - mean_from_formula(outputᵢ) * mean_from_formula(outputⱼ)

  𝔼gᵢ::Real = mean_from_formula(selection)
  𝕍gᵢ::Real = var_from_formula(selection)

  fig::Makie.Figure = Figure(size=(1000, 1000), pt_per_unit=chosen_pt_per_unit, fontsize=35)
  ##=== Calculation of Approximated Moments
  begin
    pdf_height = maximum(kde(gᵢ).density)

    ax = fig[1,1] = Makie.Axis(fig;
      limits=(mean(gᵢ) - 3.0 * std(gᵢ), mean(gᵢ) + 3.0 * std(gᵢ), 0, pdf_height + chosen_offset),
      xlabel=L"g_{%$selection}",
      # xlabel=L"g_{%$selection} = \mathcal{N}(f(μ_{%$selection} \pm \beta))",
      ylabel=L"p(g_{%$selection})")
      # title=L"p(g_{%$selection}):\text{%$num_layers Layers}, \quad |\beta| \leq %$β")
    hist!(ax, gᵢ, normalization=:pdf, bins=100, label_formatter=x -> round(x, digits=2), label_size=15, strokewidth=0.5, strokecolor=(:black, 0.5), color=:values)
    # Makie.density!(gᵢ, label="KDE Estimation")

    CairoMakie.lines!(ax, x * (π / L) .+ dot(αᵢNL, Fₙₗ(μₚ)), abs.(𝒻̂), label=L"\mathcal{F}^{-1}(\Pi_{k=1}^{%$(Nx+1)}sinc(\ell_k a_{%$selection}))", color=:darkgoldenrod, linewidth=7.5)
    mean_value = Format.format(mean(gᵢ), precision=4)
    mean_val_from_formula = Format.format(𝔼gᵢ, precision=4)
    var_value = Format.format(var(gᵢ), precision=4)
    var_val_from_formula = Format.format(𝕍gᵢ, precision=4)

    vlines!(ax, mean(gᵢ), label=L"\mathbb{E}[g_k]_{\text{num}} = %$mean_value", linestyle=:solid, linewidth=5.5)
    vlines!(ax, 𝔼gᵢ, label=L"\mathbb{E}[g_k]_{\text{formula}} = %$mean_val_from_formula", linestyle=:dash, linewidth=5.5)
    vlines!(ax, var(gᵢ), label=L"\mathbb{V}[g_k]_{\text{num}} = %$var_value", linestyle=:dash, linewidth=6.5,color=:transparent)
    vlines!(ax, 𝕍gᵢ, label=L"\mathbb{V}[g_k]_{\text{formula}} = %$var_val_from_formula", linestyle=:dash, linewidth=6.5,color=:transparent)

    # CURRENTLY BROKEN with Makie.density
    # fig[1, 2] = Legend(fig, ax, framevisible = false)

    fig
  end
  return fig
end



function plot_output_marginals_noLegend(fig::Makie.Figure, fig_idx1::Int, fig_idx2::Int, f_input::Vector;
  selection::Int=rand(1:length(f_input)),
  𝒩ₙₙNL::NeuralNet,
  𝚯_NL::Params,
  Fₙₗ::F,
  ∂𝐳Fₙₗ::D,
  β_noise::Real=1.5,
  noise_distrib::Distribution,
  chosen_offset::Real=0.5,
  chosen_pt_per_unit::Real=0.5) where {D<:Function,
  F<:Function,
  NeuralNet<:Function,Params<:ComponentVector}

  _, output_noised_g_part_samples_NL = generate_noised_samples(f_input;
    𝒩ₙₙ=𝒩ₙₙNL,
    noise_pdf=noise_distrib)

  num_nonHidden_layers::Int = 1
  num_layers::Int = length(keys(𝚯_NL)) - num_nonHidden_layers

  #===
  Recall the parameters we have:
  ===#
  # e.g. selection = 14
  # selection::Int = rand(1:length(f_input))
  #=
  Recall: Nx = Ngll_grid - 1
  =#
  Nx::Int = length(f_input) - 1
  μₚ::Vector = f_input
  𝐀_NL::Matrix = unpack_params(𝚯_NL)[1][end]
  αᵢNL::Vector = 𝐀_NL[selection, :]

  ∂𝐳Fₙₗ_ofμₚ::Matrix = ∂𝐳Fₙₗ(μₚ)
  Q::Matrix = 𝐀_NL * ∂𝐳Fₙₗ_ofμₚ
  gᵢ::Vector = output_noised_g_part_samples_NL[selection, :]

  γ::Vector = vec(transpose(αᵢNL) * ∂𝐳Fₙₗ_ofμₚ)
  ℓ::Vector = β_noise * γ

  ##=== Calculation of IFT
  N::Int = 2048
  L::Int = N / 2
  x::LinRange = LinRange(-L, L, N)

  𝒻::Vector = N * (1.0 / (2.0 * π)) * Πsinc(ℓ, x; Npoints=Nx)

  # Step 2: Perform the inverse Fourier transform
  𝒻̂ = ift(𝒻)
  ##=== Calculation of IFT

  ##=== Calculation of Approximated Moments

  ρ_out::Vector = 𝐀_NL * Fₙₗ(μₚ)

  ω_shift(xᵢ::LinRange; μ::Vector)::Vector = xᵢ .+ dot(αᵢNL, Fₙₗ(μ))
  mean_from_formula(outputᵢ::Int)::Real = dot(𝐀_NL[outputᵢ, :], Fₙₗ(μₚ))
  mean_prod(outputᵢ::Int, outputⱼ::Int)::Real = ρ_out[outputᵢ] * ρ_out[outputⱼ] + ((β_noise^2) / 3.0) * dot(Q[outputᵢ, :], Q[outputⱼ, :])
  var_from_formula(outputᵢ::Int)::Real = mean_prod(outputᵢ, outputᵢ) - mean_from_formula(outputᵢ)^2
  Cov_from_formula(outputᵢ::Int, outputⱼ::Int)::Real = mean_prod(outputᵢ, outputⱼ) - mean_from_formula(outputᵢ) * mean_from_formula(outputⱼ)

  𝔼gᵢ::Real = mean_from_formula(selection)
  𝕍gᵢ::Real = var_from_formula(selection)

  # fig::Makie.Figure = Figure(size=(1000, 1000), pt_per_unit=chosen_pt_per_unit, fontsize=35)
  ##=== Calculation of Approximated Moments
  begin
    pdf_height = maximum(kde(gᵢ).density)

    ax = Makie.Axis(fig[fig_idx1,fig_idx2];
      limits=(mean(gᵢ) - 3.0 * std(gᵢ), mean(gᵢ) + 3.0 * std(gᵢ), 0, pdf_height + chosen_offset),
      xlabel=L"g_{%$selection}",
      # xlabel=L"g_{%$selection} = \mathcal{N}(f(μ_{%$selection} \pm \beta))",
      ylabel=L"p(g_{%$selection})")
      # title=L"p(g_{%$selection}):\text{%$num_layers Layers}, \quad |\beta| \leq %$β")
    hist!(ax, gᵢ, normalization=:pdf, bins=100, label_formatter=x -> round(x, digits=2), label_size=15, strokewidth=0.5, strokecolor=(:black, 0.5), color=:values)
    # Makie.density!(gᵢ, label="KDE Estimation")

    CairoMakie.lines!(ax, x * (π / L) .+ dot(αᵢNL, Fₙₗ(μₚ)), abs.(𝒻̂), label=L"\mathcal{F}^{-1}(\Pi_{k=1}^{%$(Nx+1)}sinc(\ell_k a_{%$selection}))", color=:darkgoldenrod, linewidth=7.5)
    mean_value = Format.format(mean(gᵢ), precision=4)
    mean_val_from_formula = Format.format(𝔼gᵢ, precision=4)
    var_value = Format.format(var(gᵢ), precision=4)
    var_val_from_formula = Format.format(𝕍gᵢ, precision=4)

    vlines!(ax, mean(gᵢ), label=L"\mathbb{E}[g_k]_{\text{num}} = %$mean_value", linestyle=:solid, linewidth=5.5)
    vlines!(ax, 𝔼gᵢ, label=L"\mathbb{E}[g_k]_{\text{formula}} = %$mean_val_from_formula", linestyle=:dash, linewidth=5.5)
    vlines!(ax, var(gᵢ), label=L"\mathbb{V}[g_k]_{\text{num}} = %$var_value", linestyle=:dash, linewidth=6.5,color=:transparent)
    vlines!(ax, 𝕍gᵢ, label=L"\mathbb{V}[g_k]_{\text{formula}} = %$var_val_from_formula", linestyle=:dash, linewidth=6.5,color=:transparent)

    # CURRENTLY BROKEN with Makie.density
    # fig[1, 2] = Legend(fig, ax, framevisible = false)

    fig
  end
  return fig, ax
end


function plot_output_marginals(f_input::Vector,
  sample_indices::StepRange{<:Int},
  plot_positions::Vector{Tuple{T,T}};
  𝒩ₙₙNL::NeuralNet,
  𝚯_NL::Params,
  Fₙₗ::F,
  ∂𝐳Fₙₗ::D,
  β_noise::Real=1.5,
  noise_distrib::Distribution,
  chosen_pt_per_unit::Real = 0.5) where {T<:Int,
  D<:Function,
  F<:Function,
  NeuralNet<:Function,Params<:ComponentVector}

  _, output_noised_g_part_samples_NL = generate_noised_samples(f_input;
    𝒩ₙₙ=𝒩ₙₙNL,
    noise_pdf=noise_distrib)

  num_nonHidden_layers::Int = 1
  num_layers::Int = length(keys(𝚯_NL)) - num_nonHidden_layers

  #===
  Recall the parameters we have:
  ===#
  #=
  Recall: Nx = Ngll_grid - 1
  =#
  Nx::Int = length(f_input) - 1
  μₚ::Vector = f_input
  𝐀_NL::Matrix = unpack_params(𝚯_NL)[1][end]
  # αᵢNL::Vector = 𝐀_NL[selection, :]

  ∂𝐳Fₙₗ_ofμₚ::Matrix = ∂𝐳Fₙₗ(μₚ)
  Q::Matrix = 𝐀_NL * ∂𝐳Fₙₗ_ofμₚ

  ##=== Calculation of IFT (Values that don't depend on selectionᵢ)
  N::Int = 2048
  L::Int = N / 2
  x::LinRange = LinRange(-L, L, N)
  ##=== Calculation of IFT (Values that don't depend on selectionᵢ)

  ##=== Calculation of Approximated Moments (Formulas that are agnostic to selectionᵢ)

  ρ_out::Vector = 𝐀_NL * Fₙₗ(μₚ)

  mean_from_formula(outputᵢ::Int)::Real = dot(𝐀_NL[outputᵢ, :], Fₙₗ(μₚ))
  mean_prod(outputᵢ::Int, outputⱼ::Int)::Real = ρ_out[outputᵢ] * ρ_out[outputⱼ] + ((β_noise^2) / 3.0) * dot(Q[outputᵢ, :], Q[outputⱼ, :])
  var_from_formula(outputᵢ::Int)::Real = mean_prod(outputᵢ, outputᵢ) - mean_from_formula(outputᵢ)^2
  Cov_from_formula(outputᵢ::Int, outputⱼ::Int)::Real = mean_prod(outputᵢ, outputⱼ) - mean_from_formula(outputᵢ) * mean_from_formula(outputⱼ)
  ##=== Calculation of Approximated Moments (Formulas that are agnostic to selectionᵢ)

  fig::Makie.Figure = Figure(size=(1000, 350), pt_per_unit=chosen_pt_per_unit, fontsize=25)

  for (plot_coord, point_selection) in zip(plot_positions, sample_indices)
    row = plot_coord[1] # actually col
    col = plot_coord[2] # actually row

    selection = point_selection
    αᵢNL::Vector = 𝐀_NL[selection, :]

    gᵢ::Vector = output_noised_g_part_samples_NL[selection, :]

    γ::Vector = vec(transpose(αᵢNL) * ∂𝐳Fₙₗ_ofμₚ)
    ℓ::Vector = β_noise * γ

    𝒻::Vector = N * (1.0 / (2.0 * π)) * Πsinc(ℓ, x; Npoints=Nx)

    # Step 2: Perform the inverse Fourier transform
    𝒻̂ = ift(𝒻)
    ω_shift(xᵢ::LinRange; μ::Vector)::Vector = xᵢ .+ dot(αᵢNL, Fₙₗ(μ))
    ##=== Calculation of IFT

    ##=== Calculation of Approximated Moments
    𝔼gᵢ::Real = mean_from_formula(selection)
    𝕍gᵢ::Real = var_from_formula(selection)
    ##===

    begin
      pdf_height = maximum(kde(gᵢ).density)
      # if selection == 1
      #   ax = Makie.Axis(fig[col, row];
      #     limits=(mean(gᵢ) - 3.0 * std(gᵢ), mean(gᵢ) + 3.0 * std(gᵢ), 0, pdf_height + 1.0),
      #     xticks=[-4.0, -3.0, -2.0],
      #     # xtickformat=values -> [L"0.0", L"0.5\times 10^{-2}", L"1.0\times 10^{-2}"],
      #     xticklabelrotation=-195.25,
      #     xlabel=L"g_{%$selection}",
      #     ylabel=L"p(g_{%$selection})")
      #     # title=L"p(g_{%$selection}):\text{%$num_layers Layers}, \quad |\beta| \leq %$β")
      # else
        ax = Makie.Axis(fig[col, row];
          limits=(mean(gᵢ) - 3.0 * std(gᵢ), mean(gᵢ) + 3.0 * std(gᵢ), 0, pdf_height + 1.0),
          xlabel=L"g_{%$selection}",
          ylabel=L"p(g_{%$selection})")
          # title=L"p(g_{%$selection}):\text{%$num_layers Layers}, \quad |\beta| \leq %$β")
      # end
      hist!(ax, gᵢ, normalization=:pdf, bins=100, label_formatter=x -> round(x, digits=2), label_size=15, strokewidth=0.5, strokecolor=(:black, 0.5), color=:values)
      # Makie.density!(gᵢ, label="KDE Estimation")


      lines!(ax, ω_shift(x * (π / L); μ=μₚ), abs.(𝒻̂), label=L"\mathcal{F}^{-1}(\Pi_{k=1}^{%$(Nx+1)}sinc(\ell_k a_{%$selection}))", color=:darkgoldenrod, linewidth=3.5)
      mean_value = Format.format(mean(gᵢ), precision=4)
      mean_val_from_formula = Format.format(𝔼gᵢ, precision=4)
      var_value = Format.format(var(gᵢ), precision=4)
      var_val_from_formula = Format.format(𝕍gᵢ, precision=4)

      vlines!(ax, mean(gᵢ), label=L"\mathbb{E}[g_k]_{\text{num}} = %$mean_value", linestyle=:dash)
      vlines!(ax, 𝔼gᵢ, label=L"\mathbb{E}[g_k]_{\text{formula}} = %$mean_val_from_formula", linestyle=:dash)
      vlines!(ax, var(gᵢ), label=L"\mathbb{V}[g_k]_{\text{num}} = %$var_value", linestyle=:dash, linewidth=6.5,color=:transparent)
      vlines!(ax, 𝕍gᵢ, label=L"\mathbb{V}[g_k]_{\text{formula}} = %$var_val_from_formula", linestyle=:dash, linewidth=6.5,color=:transparent)

      # CURRENTLY BROKEN, wait for update to provide plot legend
      # fig[1, 4] = Legend(fig, ax, "Trig Functions", framevisible = false)
      fig
    end
  end
  return fig, output_noised_g_part_samples_NL

end

function plot_output_cornerPlot(f_input::Vector;
  selection::Int=rand(1:length(f_input)),
  𝒩ₙₙNL::NeuralNet,
  𝚯_NL::Params,
  Fₙₗ::F,
  ∂𝐳Fₙₗ::D,
  β_noise::Real=1.5,
  noise_distrib::Distribution,
  chosen_pt_per_unit::Real = 0.5) where {D<:Function,
  F<:Function,
  NeuralNet<:Function,Params<:ComponentVector}

  _, output_noised_g_part_samples_NL = generate_noised_samples(f_input;
    𝒩ₙₙ=𝒩ₙₙNL,
    noise_pdf=noise_distrib)

  num_nonHidden_layers::Int = 1
  num_layers::Int = length(keys(𝚯_NL)) - num_nonHidden_layers

  #===
  Recall the parameters we have:
  ===#
  # e.g. selection = 14
  # selection::Int = rand(1:length(f_input))
  #=
  Recall: Nx = Ngll_grid - 1
  =#
  Nx::Int = length(f_input) - 1
  μₚ::Vector = f_input
  𝐀_NL::Matrix = unpack_params(𝚯_NL)[1][end]
  αᵢNL::Vector = 𝐀_NL[selection, :]

  ∂𝐳Fₙₗ_ofμₚ::Matrix = ∂𝐳Fₙₗ(μₚ)
  Q::Matrix = 𝐀_NL * ∂𝐳Fₙₗ_ofμₚ
  # q_i::Vector = Q[selection, :]
  gᵢ::Vector = output_noised_g_part_samples_NL[selection, :]

  γ::Vector = vec(transpose(αᵢNL) * ∂𝐳Fₙₗ_ofμₚ)
  ℓ::Vector = β_noise * γ

  ##=== Calculation of IFT
  N::Int = 2048
  L::Int = N / 2
  x::LinRange = LinRange(-L, L, N)

  𝒻::Vector = N * (1.0 / (2.0 * π)) * Πsinc(ℓ, x; Npoints=Nx)

  # Step 2: Perform the inverse Fourier transform
  𝒻̂ = ift(𝒻)
  ##=== Calculation of IFT

  ##=== Calculation of Approximated Moments

  ρ_out::Vector = 𝐀_NL * Fₙₗ(μₚ)

  ω_shift(xᵢ::LinRange; μ::Vector)::Vector = xᵢ .+ dot(αᵢNL, Fₙₗ(μ))
  mean_from_formula(outputᵢ::Int)::Real = dot(𝐀_NL[outputᵢ, :], Fₙₗ(μₚ))
  mean_prod(outputᵢ::Int, outputⱼ::Int)::Real = ρ_out[outputᵢ] * ρ_out[outputⱼ] + ((β_noise^2) / 3.0) * dot(Q[outputᵢ, :], Q[outputⱼ, :])
  var_from_formula(outputᵢ::Int)::Real = mean_prod(outputᵢ, outputᵢ) - mean_from_formula(outputᵢ)^2
  Cov_from_formula(outputᵢ::Int, outputⱼ::Int)::Real = mean_prod(outputᵢ, outputⱼ) - mean_from_formula(outputᵢ) * mean_from_formula(outputⱼ)

  𝔼gᵢ::Real = mean_from_formula(selection)
  𝕍gᵢ::Real = var_from_formula(selection)

  ##=== Calculation of Approximated Moments

  # pairplot_fig::Makie.Figure = Figure(size=(1000, 1050), pt_per_unit=chosen_pt_per_unit, fontsize=20)
  pairplot_fig::Makie.Figure = Figure(size=(1000, 1000), pt_per_unit=chosen_pt_per_unit, fontsize=20)

  c1 = Makie.wong_colors(0.5)[1]
  c2 = Makie.wong_colors(0.5)[2]
  c3 = Makie.wong_colors(0.5)[3]
  c4 = Makie.wong_colors(0.5)[4]
  c5 = Makie.wong_colors(0.5)[5]

  df2::DataFrame = DataFrame(transpose(output_noised_g_part_samples_NL), :auto);

  pairplot_ax = Makie.Axis(pairplot_fig[1,1])

  df_select = DataFrame(g₁=df2[:, 1], g₆=df2[:, 6], g₁₁=df2[:, 11], g₁₆=df2[:, 16], g₂₁=df2[:, 21]);

  pairplot(pairplot_fig[1, 1],
  # pairplot_ax,
  df_select => (
    PairPlots.HexBin(colormap=Makie.cgrad([:transparent, c1]),),
    PairPlots.Scatter(filtersigma=2, color=c1),
    PairPlots.Contour(),
    # New:
    PairPlots.MarginHist(color=c1),
    # PairPlots.MarginDensity(color=c4),
    PairPlots.Truth(
      (;
        g₁=[mean_from_formula(1)],
        g₆=[mean_from_formula(6)],
        g₁₁=[mean_from_formula(11)],
        g₁₆=[mean_from_formula(16)],
        g₂₁=[mean_from_formula(21)],
      ),
      label=L"\mathbb{E}[g_k]_{\text{formula}}",
      color=c3,
      fontsize=20
    ),
    # PairPlots.TrendLine(color=c2, label="Trendline"),
    # PairPlots.Correlation(),
    PairPlots.PearsonCorrelation(position=Makie.Point2f(0.2, 0.2), fontsize=18),
    # bottomleft=false, topright=true
  ),
  # axis=pairplot_ax,
  bottomleft=false, topright=true
)

# pairplot_fig[1, 1] = Legend(pairplot_fig, pairplot_ax, framevisible = false)

pairplot_fig

end