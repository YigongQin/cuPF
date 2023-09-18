import numpy as np
import h5py
import sys, os
from graph_datastruct import graph
from math import pi
Dh = 8.43e7                     # heat diffusion
c_infty = 3
m_slope = 2.6                    # liquidus slope K/wt    
GT = 0.347                       # Gibbs-Thompson coefficient K*um
k = 0.14                        # interface solute partition coefficient
Dl = 3000                       # liquid diffusion coefficient      um**2/s

L_cp = 229
delta = 0.01                    # strength of the surface tension anisotropy         
kin_delta = 0.11
beta0 = 1e-7                    # linear coefficient

mu_k = 0.217e6                         #um/s/K
Tmelt = 933.3
underCoolingRate = 10

eps = 1e-8                       # divide-by-zero treatment
alpha0 = 0                       # misorientation angle in degree
U0 = -1                    # initial value for U, -1 < U0 < 0
ictype = 0                    # initial condtion: 0 for semi-circular, 1 for planar interface, 2 for sum of sines

## MPI
haloWidth = 1;
xmin = 0
ymin = 0
zmin = 0


# nuleation parameters
undcool_mean = 2   # Kelvin  nuleantion barrier
undcool_std = 0.5   # Kelvin fixed
nuc_Nmax = 0.05      # 1/um^2 density; 0 to very big number 
nuc_rad = 0.3      # 0.2 um radius of a nucleai

## noise
eta = 0.0  
noi_period = 200


# simulation parameters
dx = 0.8                            # mesh width
W0 = 0.1                    # interface thickness      um
cfl = 1.5
asp_ratio_yx = 1
asp_ratio_zx = 1                    # aspect ratio
moving_ratio = 0.2
nts = 1          # number snapshots to save, Mt/nts must be int
Lx = 20

nx = 13
ny = 13
nz = 13
nt = 11
Ly = Lx*asp_ratio_yx
Lz = Lx*asp_ratio_zx
BC = Lx/(nx-3) 
top = 10


seed = int(sys.argv[1])
''' for test use seed > 10000'''
if seed<10000:  
   ''' grid sampling'''  
   G_list = np.linspace(10, 0.5, 39)
   R_list = np.linspace(2, 0.2, 37)
   
   Gid = seed%len(G_list)
   Rid = seed//len(G_list)
   G = G_list[Gid]
   Rmax = 1e6*R_list[Rid]

   print('samples in G, R domain: ', len(G_list), len(R_list))

else:
   np.random.seed(seed)
   G = np.random.random()*(10-0.5) + 0.5
   Rmax = np.random.random()*(2-0.2) + 0.2
   Rmax *= 1e6

if seed == 54320:
    G = 10
    Rmax = 2e6

if seed == 0:
    G = 10
    Rmax = 20e5

print('sampled G, R values: ', G, Rmax)

z0 = 2

x = np.linspace(0-BC,Lx+BC,nx)
y = np.linspace(0-BC,Ly+BC,ny)
z = np.linspace(0-BC,Lz+BC,nz)
t = np.linspace(0,top/Rmax,nt)
tmax = t[-1]

T = np.zeros(nx*ny*nz*nt)
psi = np.zeros(nx*ny*nz)
U = np.zeros(nx*ny*nz)


'''create graph'''


g1 = graph(lxd = Lx, seed = seed) 
print('input shape of alpha_field, ', g1.alpha_field.shape)
rot = 0 # rotation
alpha = g1.alpha_field
alpha = np.rot90(alpha, rot)
NG = len(g1.regions)
NN = len(g1.vertices)
print('no. nodes', NN, 'no. regions', NG)
theta = np.hstack([0, g1.theta_x[1:]-rot*pi/2, g1.theta_z[1:]])

for i in range(nx*ny*nz*nt):
    
    xi = i%nx
    yi = int( (i%(nx*ny))/nx )
    zi = int( (i%(nx*ny*nz))/(nx*ny) )
    ti = int(i/(nx*ny*nz))
    
    #T[i] = 920 + G*( y[yi] - 0.5*Rmax*(t[ti]**2/tmax) - y0)
    #T[i] = Tmelt - c_infty*m_slope + G*( z[zi] - Rmax*t[ti] - z0)    
    T[i] = Tmelt - c_infty*m_slope - underCoolingRate*1e6*t[ti]
    if i==nx*ny*nz*nt-1: print(T[i], G, z[zi], Rmax, t[ti], z0)
    if ti==0:
       psi[i] = -10 #z0 - z[zi]      
        
mac_folder = './forcing/case' + str(seed) + '/'

isExist = os.path.exists(mac_folder)

if not isExist:
  
  # Create a new directory because it does not exist 
  os.makedirs(mac_folder)
  
np.savetxt(mac_folder+'x.txt', x, fmt='%1.4e',delimiter='\n') 
np.savetxt(mac_folder+'y.txt', y, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'z.txt', z, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'t.txt', t, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'psi.txt', psi, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'U.txt', U, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'alpha.txt', alpha, fmt='%d',delimiter='\n')
np.savetxt(mac_folder+'theta.txt', theta, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'G.txt', np.asarray([G]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'Rmax.txt', np.asarray([Rmax*1e-6]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'NG.txt', np.asarray([NG]), fmt='%d',delimiter='\n')
np.savetxt(mac_folder+'NN.txt', np.asarray([NN]), fmt='%d',delimiter='\n')
np.savetxt(mac_folder+'z0.txt', np.asarray([z0]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'top.txt', np.asarray([top]), fmt='%1.4e',delimiter='\n')

hf = h5py.File(mac_folder+'Temp.h5', 'w')
hf.create_dataset('Temp', data=T)
hf.close()
