# constant physical parameters
Tmelt = 933.3
Dh = 8.43e7                     # heat diffusion
L_cp = 229
GT = 0.347                      # Gibbs-Thompson coefficient [K*um]
beta0 = 1e-7                    # linear coefficient
mu_k = 0.217e6                  # [um/s/K]
delta = 0.01                    # strength of the surface tension anisotropy         
kin_delta = 0.11                # kinetic anisotropy


# nuleation parameters
undcool_mean = 2                # Kelvin  nuleantion barrier
undcool_std = 0.5               # Kelvin fixed
nuc_Nmax = 0.01                 # density [1/um^2] 
nuc_rad = 0.4                   # radius of a nucleai

# macro grid parameters
nx = 83
ny = 83
nz = 43
nt = 5


## MPI
haloWidth = 1
xmin = 0
ymin = 0
zmin = 0

## noise
eta = 0.0  
noi_period = 200
eps = 1e-8                      # divide-by-zero treatment
ictype = 0                      # initial condtion

# simulation parameters
dx = 0.8                        # mesh width
W0 = 0.15625                        # interface thickness [um]
cfl = 0.6                       # cfl number
asp_ratio_yx = 1             # aspect ratio of domain z/x
asp_ratio_zx = 0.5             # aspect ratio of domain z/x
moving_ratio = 0.2
nts = 1                         # number snapshots to save, Mt/nts must be int
Lx = 80
Ly = Lx*asp_ratio_yx
Lz = Lx*asp_ratio_zx
BC = Lx/(nx-3) 
track = Lx-4
top = track
z0 = 0
#r0 = Lz

G = 5
Rmax = 2e6
underCoolingRate = 10
V = 4e6

# initial liquid param
underCoolingRate0 = 20
nuc_Nmax0 = 0.01
preMt = 4000
