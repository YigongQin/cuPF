# constant physical parameters
Tmelt = 933.3
Dh = 8.43e7                     # heat diffusion
L_cp = 229
GT = 0.347                      # Gibbs-Thompson coefficient K*um
beta0 = 1e-7                    # linear coefficient
mu_k = 0.217e6                  # [um/s/K]
delta = 0.01                    # strength of the surface tension anisotropy
kin_delta = 0.11


# nuleation parameters
undcool_mean = 2                # Kelvin  nuleantion barrier
undcool_std = 0.5               # Kelvin fixed
nuc_Nmax = 0.01                 # 1/um^2 density; 0 to very big number
nuc_rad = 0.4                   # radius of a nucleai

# macro grid parameters
nx = 13
ny = 13
nz = 23
nt = 11


## MPI
haloWidth = 1
xmin = 0
ymin = 0
zmin = 0

## noise
eta = 0.0
noi_period = 200
eps = 1e-8                       # divide-by-zero treatment
ictype = 0                    # initial condtion

# simulation parameters
dx = 0.8                            # mesh width
W0 = 0.1                    # interface thickness      um
cfl = 1.5
asp_ratio_yx = 1
asp_ratio_zx = 2                    # aspect ratio
moving_ratio = 0.2
nts = 20          # number snapshots to save, Mt/nts must be int
Lx = 40
Ly = Lx*asp_ratio_yx
Lz = Lx*asp_ratio_zx
BC = Lx/(nx-3)
top = 48
z0 = 2
r0 = 0.9*Lz

G = 10
Rmax = 2e6
underCoolingRate = 20

# initial liquid param
underCoolingRate0 = 20
nuc_Nmax0 = 0.01
preMt = 0
