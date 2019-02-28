using NetCDF

"""Gradient function. 2nd-order centred."""
function ∂x!(du::AbstractVector,u::AbstractVector)
    m = length(du)
    @boundscheck m+1 == length(u) || throw(BoundsError())

    @inbounds for i ∈ 1:m
        du[i] = one_over_dx*(u[i+1] - u[i])
    end
end

"""2-point interpolation."""
function Ix!(du::AbstractVector,u::AbstractVector)
    m = length(du)
    @boundscheck m+1 == length(u) || throw(BoundsError())

    @inbounds for i ∈ 1:m
        du[i] = 0.5*(u[i] + u[i+1])
    end
end


# Forcing
F(x,t) = -F0*sin.(8π*x/Lx .+ 200*t).*sin.(π*x/Lx).^2/rho

"""Computes the right-hand side of
        ∂u/∂t = -u*∂u/∂x -g*∂η/∂x + ν∂²u/∂x² + Fx
        ∂η/∂t = -∂(uh)/∂x. """
function rhs!(du,dη,u,η,h,u²,h_u,u²_h,dudx,p,dpdx,U,dUdx,t)

    @views h .= η .+ H      # layer thickness
    @views u² .= u.^2

    Ix!(h_u,h)
    Ix!(u²_h,u²)
    ∂x!(dudx,u)

    # Bernoulli potential + stress "tensor"
    @views p .= -.5*u²_h .- g*η[2:end] .+ ν*dudx

    # momentum equation
    ∂x!(dpdx,p)
    @views du[2:end-1] .= dpdx .+ F(x_u,t)./h_u[2:end]

    # continuity equation
    @views U .= u[1:end-1].*h_u         # volume flux
    ∂x!(dUdx,U)
    @views dη[2:end-1] .= -dUdx
end

"""Computes the right-hand side of
        ∂u/∂t = -g*∂η/∂x + ν∂²u/∂x² + Fx
        ∂η/∂t = -∂(uh)/∂x. """
function rhs_lin!(du,dη,u,η,h,u²,h_u,u²_h,dudx,p,dpdx,U,dUdx,t)

    @views h .= η .+ H      # layer thickness

    Ix!(h_u,h)
    ∂x!(dudx,u)

    # Bernoulli potential + stress "tensor"
    @views p .= -g*η[2:end] .+ ν*dudx

    # momentum equation
    ∂x!(dpdx,p)
    @views du[2:end-1] .= dpdx .+ F(x_u,t)./h_u[2:end]

    # continuity equation
    @views U .= H*u[1:end-1]         # volume flux
    ∂x!(dUdx,U)
    @views dη[2:end-1] .= -dUdx
end

"""Add halo to the left and right of each subdomain for ghost point communication."""
function add_halo(u::AbstractVector,η::AbstractVector)
    u = cat(0.,u,0.,dims=1)
    η = cat(0.,η,0.,dims=1)

    ghost_points!(u,η)
    return u,η
end

"""Propagate the boundary conditions (BC) via ghost point communication.
For multiple processes, BC of subdomains are communicated via MPI."""
function ghost_points!(u::AbstractVector,η::AbstractVector)
    # sequential, periodic BC
    u[1] = u[end-1]
    u[end] = u[2]

    η[1] = η[end-1]
    η[end] = η[2]
end

"""Preallocate intermediate variables used in the right-hand side computation."""
function preallocate(u::AbstractVector,η::AbstractVector)
    # u, η already have the halo
    u0,η0 = zero(u),zero(η)
    u1,η1 = zero(u),zero(η)
    du,dη = zero(u),zero(η)

    u²,h = zero(u),zero(η)
    p,dpdx = zeros(length(η)-1),zeros(length(η)-2)

    # each centred gradient/interpolation removes one grid point
    h_u = zeros(length(η)-1)
    u²_h = zeros(length(u)-1)
    dudx = zeros(length(u)-1)

    U = zeros(length(u)-1)
    dUdx = zeros(length(u)-2)

    return u0,η0,u1,η1,du,dη,h,u²,h_u,u²_h,dudx,p,dpdx,U,dUdx
end

"""4th order Runge-Kutta time integration."""
function time_integration(Nt,u,η)
    # pre-allocate memory
    u,η = add_halo(u,η)
    u0,η0,u1,η1,du,dη,h,u²,h_u,u²_h,dudx,p,dpdx,U,dUdx = preallocate(u,η)

    t = 0.

    # for output
    u_out = zeros(Nt+1,length(u)-2)     # -2 to remove ghost points
    η_out = zeros(Nt+1,length(η)-2)

    # store the initial conditions
    u_out[1,:] = u[2:end-1]
    η_out[1,:] = η[2:end-1]

    for i = 1:Nt

        ghost_points!(u,η)
        u1 .= u
        η1 .= η

        for RKi = 1:4
            if RKi > 1
                ghost_points!(u1,η1)
            end

            if size(ARGS)[1] > 0 && ARGS[1] == "lin"
                rhs_lin!(du,dη,u1,η1,h,u²,h_u,u²_h,dudx,p,dpdx,U,dUdx,t)
            else
                rhs!(du,dη,u1,η1,h,u²,h_u,u²_h,dudx,p,dpdx,U,dUdx,t)
            end

            if RKi < 4 # RHS update for the next RK-step
                u1 .= u .+ RKβ[RKi]*dt*du
                η1 .= η .+ RKβ[RKi]*dt*dη
            end

            # Summing all the RHS on the go
            u0 .+= RKα[RKi]*dt*du
            η0 .+= RKα[RKi]*dt*dη

        end

        u .= u0
        η .= η0
        t += dt

        # store for output
        u_out[i+1,:] .= u[2:end-1]
        η_out[i+1,:] .= η[2:end-1]

    end
    u_out,η_out
end

function setup_output(filename, num_points)
    # Define dimensions
    timedim = NcDim("time", 0, unlimited=true)
    idim    = NcDim("i", num_points)

    # Define dimension variables
    timevar = NcVar("time", timedim, t=Int32)
    ivar    = NcVar("i",    idim,    t=Int32)

    # Define multidimensional variables
    u   = NcVar("u", [idim, timedim], t=Float32)
    η   = NcVar("eta", [idim, timedim], t=Float32)

    nc = NetCDF.create(filename, [u, η, timevar])
end

function output(ncfile, t_index, t, u, η)
    println("Writing timestep $t_index")
    NetCDF.putvar(ncfile, "u", u, start=[1,t_index], count=[-1,1])
    NetCDF.putvar(ncfile, "eta", η, start=[1,t_index], count=[-1,1])
end

const g = 10.
const H = 10.
const N = 500
const Nt = 2000
const cfl = 0.9
const rho = 1.
const ν = 0.006
const F0 = 500.
const Lx = 1.
const dx = Lx/N
const one_over_dx = 1/dx
const cph = sqrt(g*H)
const dt = cfl * dx / cph
const RKα = [1/6.,1/3.,1/3.,1/6.]
const RKβ = [0.5,0.5,1.]

if size(ARGS)[1] > 0 && ARGS[1] == "lin"
    output_file = "training_data_lin.nc"
else
    output_file = "training_data.nc"
end

# grid
const x_h = dx/2:dx:Lx
const x_u = dx:dx:Lx        # no u-point at x=0 but at x=L (periodicity)

# initial conditions
u_ini = fill(0.,N)
η_ini = fill(0.,N)

u,η = time_integration(Nt,u_ini,η_ini)

nc = setup_output(output_file, N)
for i = 1:Nt+1
    output(nc, i, i, u[i,:], η[i,:])
end