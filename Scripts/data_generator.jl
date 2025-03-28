include("Discretizations/gll_discretize.jl")

using .GllDiscretize: Discretization, generate_gll_disc

function generate_data( Ngrids_x::Int,
                        Ngrids_y::Int,
                        func_in::Function,
                        operator::Function,
                        NSAMP::Int = 10000)
  # Number of sample input/output curves

  disc_x = generate_gll_disc(Ngrids_x, -1.0, 1.0);
  disc_y = generate_gll_disc(Ngrids_y, -1.0, 1.0);

  x = disc_x.nodes;
  y = disc_y.nodes;
  INPUT = zeros(Ngrids_x+1, NSAMP);
  OUTPUT = zeros(Ngrids_y+1,NSAMP);

  # Threads.@threads for ii in 1:NSAMP
  for ii in 1:NSAMP
    f = func_in(x);

    INPUT[:,ii] = f;
    OUTPUT[:,ii] = operator(f, y, disc_x);
  end

  return vec(x), vec(y), INPUT, OUTPUT
end

# When we are only working with 1 sample:
function generate_data( Ngrids_x::Int,
  Ngrids_y::Int,
  func_in::Function,
  operator::Function)
  # NSAMP::Int = 10000)
  # Number of sample input/output curves

  disc_x = generate_gll_disc(Ngrids_x, -1.0, 1.0);
  disc_y = generate_gll_disc(Ngrids_y, -1.0, 1.0);

  x = disc_x.nodes;
  y = disc_y.nodes;
  INPUT = zeros(Ngrids_x+1);
  OUTPUT = zeros(Ngrids_y+1);

  # Threads.@threads for ii in 1:NSAMP
  # for ii in 1:NSAMP
  f = func_in(x);

  INPUT[:] = func_in(x);
  OUTPUT[:] = operator(f, y, disc_x);
  # end

  return vec(x), vec(y), INPUT, OUTPUT
end






function generate_data_zip( Ngrids_x::Int,
  Ngrids_y::Int,
  func_in::Function,
  operator::Function,
  NSAMP::Int = 10000)
# Number of sample input/output curves

disc_x = generate_gll_disc(Ngrids_x, -1.0, 1.0);
disc_y = generate_gll_disc(Ngrids_y, -1.0, 1.0);

x = disc_x.nodes;
y = disc_y.nodes;
# INPUT = [func_in(x) for _ in 1:NSAM];
# OUTPUT = zeros(Ngrids_y+1,NSAMP);
INPUT =[];
# INPUT = Vector{Vector{Float32}}();
OUTPUT = [];

# Threads.@threads for ii in 1:NSAMP
for ii in 1:NSAMP
f = func_in(x);

# INPUT[:,ii] = f;
push!(INPUT, f);
# OUTPUT[:,ii] = operator(f, y, disc_x);
push!(OUTPUT, operator(f, y, disc_x));
end

return vec(x), vec(y), INPUT, OUTPUT
end

# When we are only working with 1 sample:
function generate_data_zip( Ngrids_x::Int,
Ngrids_y::Int,
func_in::Function,
operator::Function)
# NSAMP::Int = 10000)
# Number of sample input/output curves

disc_x = generate_gll_disc(Ngrids_x, -1.0, 1.0);
disc_y = generate_gll_disc(Ngrids_y, -1.0, 1.0);

x = disc_x.nodes;
y = disc_y.nodes;
INPUT = zeros(Ngrids_x+1);
OUTPUT = zeros(Ngrids_y+1);

# Threads.@threads for ii in 1:NSAMP
# for ii in 1:NSAMP
f = func_in(x);

INPUT[:] = func_in(x);
OUTPUT[:] = operator(f, y, disc_x);
# end

return vec(x), vec(y), INPUT, OUTPUT
end