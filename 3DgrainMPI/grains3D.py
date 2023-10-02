import numpy as np
import h5py
import sys, os
from graph_datastruct import graph
from math import pi
import argparse
from TemperatureProfile3DAnalytic import ThermalProfile

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
nx = 43
ny = 43
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
eps = 1e-8                       # divide-by-zero treatment
ictype = 0                    # initial condtion

# simulation parameters
dx = 0.8                            # mesh width
W0 = 0.1                    # interface thickness      um
cfl = 1.0
asp_ratio_yx = 1
asp_ratio_zx = 0.5                    # aspect ratio
moving_ratio = 0.2
nts = 1          # number snapshots to save, Mt/nts must be int
Lx = 20
Ly = Lx*asp_ratio_yx
Lz = Lx*asp_ratio_zx
BC = Lx/(nx-3) 
top = 10
z0 = 1
r0 = 0.9*Lz

G = 10
Rmax = 2e6
underCoolingRate = 20


# initial liquid param
underCoolingRate0 = 20
nuc_Nmax0 = 0.01
preMt = 2000


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Generate thermal input for PF")
    parser.add_argument("--outfile_folder", type=str, default = '/scratch1/07428/ygqin/graph/')
    parser.add_argument("--mode", type=str, default = 'check')
    parser.add_argument("--seed", type=int, default = 1)
    parser.add_argument("--save3Ddata", type=int, default = 0)
    parser.add_argument("--meltpool", type=str, default = 'cylinder')
    parser.add_argument("--boundary", type=str, default = '000')
    parser.add_argument("--mpi", type=int, default = 1)

    parser.add_argument("--nucleation", dest='nucl', action='store_true')
    parser.set_defaults(nucl=False)

    parser.add_argument("--liquidStart", dest='liquid', action='store_true')
    parser.set_defaults(liquid=False)

    parser.add_argument("--lineConfig", dest='line', action='store_true')
    parser.set_defaults(line=False)
    
    parser.add_argument("--nuclGridSampl", dest='nuclGrid', action='store_true')
    parser.set_defaults(nuclGrid=False)

    parser.add_argument("--grGridSampl", dest='grGrid', action='store_true')
    parser.set_defaults(grGrid=False)
    

    
    args = parser.parse_args()     
    

    seed = args.seed
    
    if args.nuclGrid:
        if seed<10000:  
           ''' grid sampling'''  
           UC_list = np.linspace(5, 50, 10)
           Nmax_list = np.linspace(0.001, 0.01, 10)
           
           Uid = seed%len(UC_list)
           Nid = seed//len(Nmax_list)
           underCoolingRate = UC_list[Uid]
           nuc_Nmax = Nmax_list[Nid]
        
    if G>0:
       Rmax = underCoolingRate*1e6/G
       
       
    print('sampled undercooling, G, R values: ', underCoolingRate, G, Rmax)
    print('nucleation density, radius: ', nuc_Nmax, nuc_rad)
    

    
    
    '''create a planar graph'''
    
    g1 = graph(lxd = Lx, seed = seed) 
    print('input shape of alpha_field, ', g1.alpha_field.shape)
    
    alpha = g1.alpha_field
    
    # create nucleation pool
    num_nucleatioon_theta = 1000
    
    # sample orientations
    ux = np.random.randn(num_nucleatioon_theta)
    uy = np.random.randn(num_nucleatioon_theta)
    uz = np.random.randn(num_nucleatioon_theta)
    
    u = np.sqrt(ux**2+uy**2+uz**2)
    ux = ux/u
    uy = uy/u
    uz = uz/u
    
    theta_x = np.zeros(1 + num_nucleatioon_theta)
    theta_z = np.zeros(1 + num_nucleatioon_theta)
    theta_x[1:] = np.arctan2(uy, ux)%(pi/2)
    theta_z[1:] = np.arctan2(np.sqrt(ux**2+uy**2), uz)%(pi/2)
    
    NG = len(g1.regions) + num_nucleatioon_theta
    NN = len(g1.vertices)
    
    print('no. nodes', NN, 'no. regions', NG)
    
    theta = np.hstack([0, g1.theta_x[1:], theta_x[1:], g1.theta_z[1:], theta_z[1:]])
    
  
    
    x = np.linspace(0-BC,Lx+BC,nx)
    y = np.linspace(0-BC,Ly+BC,ny)
    z = np.linspace(0-BC,Lz+BC,nz)
    

    t = np.linspace(0,top/Rmax,nt)
    
    tmax = t[-1]
    
    T = np.zeros(nx*ny*nz*nt)
    psi = np.zeros(nx*ny*nz)
    U = np.zeros(nx*ny*nz)
    
    therm= ThermalProfile([Lx, Ly, Lz], [G, Rmax, underCoolingRate])
    
    for i in range(nx*ny*nz*nt):
        
        xi = i%nx
        yi = int( (i%(nx*ny))/nx )
        zi = int( (i%(nx*ny*nz))/(nx*ny) )
        ti = int(i/(nx*ny*nz))
    
    
        T[i] = therm.pointwiseTempConstGR(args.meltpool, x[xi], y[yi], z[zi], t[ti], z0=z0, r0=r0)

        if ti==0:
            
           psi[i] = therm.dist2Interface(args.meltpool, x[xi], y[yi], z[zi], z0=z0, r0=r0)  
    
    T3d0 = T[:nx*ny*nz].reshape(nx,ny,nz, order='F')
    psi3d = psi.reshape(nx,ny,nz, order='F')       
    
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
    np.savetxt(mac_folder+'Nmax.txt', np.asarray([nuc_Nmax]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'UC.txt', np.asarray([underCoolingRate]), fmt='%1.4e',delimiter='\n')
    
    np.savetxt(mac_folder+'NG.txt', np.asarray([NG]), fmt='%d',delimiter='\n')
    np.savetxt(mac_folder+'NN.txt', np.asarray([NN]), fmt='%d',delimiter='\n')
    np.savetxt(mac_folder+'z0.txt', np.asarray([z0]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'top.txt', np.asarray([top]), fmt='%1.4e',delimiter='\n')
    
    hf = h5py.File(mac_folder+'Temp.h5', 'w')
    hf.create_dataset('Temp', data=T)
    hf.close()
    
    cmd = "./phase_field grains3D.py" + " -s " + str(args.seed) + " -b " + args.boundary + " -o " + args.outfile_folder
    if args.liquid:
        cmd = cmd + " -n 1"
    elif args.nucl:
        cmd = cmd + " -n 2"
    else:
        cmd = cmd
        
    if args.mpi>1:
        cmd = "ibrun -n " + str(args.mpi) + " " + cmd
    
    print(cmd)
    
    os.system(cmd)  
