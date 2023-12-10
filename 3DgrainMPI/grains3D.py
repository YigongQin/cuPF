import numpy as np
import h5py
import sys, os
from graph_datastruct import graph
from math import pi
import argparse
from TemperatureProfile3DAnalytic import ThermalProfile
from matplotlib import pyplot as plt

def UN_sampling(seed):

    ''' grid sampling'''  
    UC_list = np.linspace(5, 50, 10)
    Nmax_list = np.linspace(0.001, 0.01, 10)
    
    Uid = seed%len(UC_list)
    Nid = seed//len(Nmax_list)
    underCoolingRate = UC_list[Uid]
    nuc_Nmax = Nmax_list[Nid]
           
    return underCoolingRate, nuc_Nmax


def UN_random_sampling(seed):
    
    np.random.seed(seed)
    underCoolingRate = np.random.random()*(50-5) + 5
    nuc_Nmax = np.random.random()*(0.01-0.001) + 0.001

    return underCoolingRate, nuc_Nmax

def GR_sampling(seed):
    
    G_list = np.linspace(10, 0.5, 39)
    R_list = np.linspace(2, 0.2, 37)
    
    Gid = seed%len(G_list)
    Rid = seed//len(G_list)
    G = G_list[Gid]
    Rmax = 1e6*R_list[Rid]
    
    return G, Rmax


def GR_random_sampling(seed):

    np.random.seed(seed)
    G = np.random.random()*(10-0.5) + 0.5
    Rmax = np.random.random()*(2-0.2) + 0.2
    Rmax *= 1e6

    return G, Rmax

def GR_constant_sampling(seed):
    
    case = (seed//10)%5
    if case == 0:
        G, R = 5, 1
    elif case == 1:
        G, R = 1, 0.4
    elif case == 2:
        G, R = 1, 1.6
    elif case == 3:
        G, R = 8, 0.4
    elif case == 4:
        G, R = 8, 1.6
    
    return G, R*1e6
    
    
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Generate thermal input for PF")
    parser.add_argument("--outfile_folder", type=str, default = '/scratch/07428/ygqin/graph/cylinder/')
    parser.add_argument("--mode", type=str, default = 'check')
    parser.add_argument("--seed", type=int, default = 10075)
    parser.add_argument("--save3Ddata", type=int, default = 0)
    parser.add_argument("--meltpool", type=str, default = 'lineTemporal')
    parser.add_argument("--boundary", type=str, default = '000')
    parser.add_argument("--mpi", type=int, default = 1)
    parser.add_argument("--gr_sample", type=int, default = 0)
    parser.add_argument("--nucl_sample", type=int, default = 0)
    parser.add_argument("--cylinder_scale_param", type=int, default = 0)

    parser.add_argument("--nucleation", dest='nucl', action='store_true')
    parser.set_defaults(nucl=False)

    parser.add_argument("--liquidStart", dest='liquid', action='store_true')
    parser.set_defaults(liquid=False)

    parser.add_argument("--lineConfig", dest='line', action='store_true')
    parser.set_defaults(line=False)
    
    
    args = parser.parse_args()     
    

    seed = args.seed
    
    if args.meltpool == 'cylinder':
        from cylinder import *
    
    if args.meltpool == 'line':
        from line import *    

    if args.meltpool == 'lineTemporal':
        from lineTemporal import *   
        

    if args.gr_sample>0:
        if args.gr_sample == 1:
            G, Rmax = GR_sampling(seed)
        elif args.gr_sample == 2:
            G, Rmax = GR_random_sampling(seed)
        elif args.gr_sample == 3:
            G, Rmax = GR_constant_sampling(seed)
        underCoolingRate = G*Rmax/1e6 

    if args.nucl_sample>0:
        if args.nucl_sample == 1:
            underCoolingRate, nuc_Nmax = UN_sampling(seed)
        elif args.nucl_sample == 2:
            underCoolingRate, nuc_Nmax = UN_random_sampling(seed)
        Rmax = underCoolingRate*1e6/G

    if args.meltpool == 'cylinder':
        if args.cylinder_scale_param == 1:
            z0 = Lz - Lz/10*(seed%10)
        if args.cylinder_scale_param == 2:
            Ly = Ly/10*(seed%10)
            Lz = Ly/2
    
    
    '''create a planar graph'''
    bc = 'periodic' if (args.boundary)[:2] == '11' else 'noflux'
    g1 = graph(lxd = Lx, seed = seed, BC = bc) 
    print('input shape of alpha_field, ', g1.alpha_field.shape)
    
    alpha = g1.alpha_field

    NG = len(g1.regions) 
    NN = len(g1.vertices)
    
    print('no. nodes', NN, 'no. regions', NG)
    
    if args.liquid or args.nucl:

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

        NG += num_nucleatioon_theta
        theta = np.hstack([0, g1.theta_x[1:], theta_x[1:], g1.theta_z[1:], theta_z[1:]])

    else:
        theta = np.hstack([0, g1.theta_x[1:], g1.theta_z[1:]])
    
  
    
    x = np.linspace(0-BC,Lx+BC,nx)
    y = np.linspace(0-BC,Ly+BC,ny)
    z = np.linspace(0-BC,Lz+BC,nz)
    
    therm = ThermalProfile([Lx, Ly, Lz], [G, Rmax, underCoolingRate], seed=seed)
    
    if args.meltpool == 'lineTemporal':
        minR = 0.2*1e6
        t_end = top/minR     # make sure at t_end the interface will reach top (has travelled distance=top)
        t = np.linspace(0, t_end, nt)
        print(t_end) 
        np.random.seed(seed)
        G_rand, R_rand = therm.RandGR(t, t_end, 2**(seed%10))
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        travelled = np.zeros(len(R_rand))
        travelled[1:] = 0.5*(R_rand[1:]+R_rand[:-1])*(t[1]-t[0])*1e6
        travelled = z0 + np.expand_dims(np.cumsum(travelled), axis=(0,1,2))
        T = np.expand_dims(G_rand, axis=(0,1,2)) * (np.expand_dims(zz, axis=-1) - travelled) 
        Ttemp = T
        psi = z0 - zz
        plt.plot(G_rand)
        plt.plot(R_rand)
        T = T.reshape((nx*ny*nz*nt), order='F')
        psi = psi.reshape((nx*ny*nz), order='F')
        G = np.mean(G_rand)
        Rmax = np.mean(R_rand)*1e6

    else:

        t_end = top/Rmax     
        t = np.linspace(0, t_end, nt)

        T = np.zeros(nx*ny*nz*nt)
        psi = np.zeros(nx*ny*nz)
        U = np.zeros(nx*ny*nz)
        
        for i in range(nx*ny*nz*nt):
            
            xi = i%nx
            yi = int( (i%(nx*ny))/nx )
            zi = int( (i%(nx*ny*nz))/(nx*ny) )
            ti = int(i/(nx*ny*nz))
        
        
            T[i] = therm.pointwiseTempConstGR(args.meltpool, x[xi], y[yi], z[zi], t[ti], z0=z0, r0=r0, angle = angle)
    
            if ti==0:
                
               psi[i] = therm.dist2Interface(args.meltpool, x[xi], y[yi], z[zi], z0=z0, r0=r0, angle = angle)  
    
        T3d0 = T[:nx*ny*nz].reshape(nx,ny,nz, order='F')
        psi3d = psi.reshape(nx,ny,nz, order='F')       

    print('sampled undercooling, G, R values: ', underCoolingRate, G, Rmax)
    print('nucleation density, radius: ', nuc_Nmax, nuc_rad)
    
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
   # np.savetxt(mac_folder+'U.txt', U, fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'alpha.txt', alpha, fmt='%d',delimiter='\n')
    np.savetxt(mac_folder+'theta.txt', theta, fmt='%1.4e',delimiter='\n')
    
    np.savetxt(mac_folder+'G.txt', np.asarray([G]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'Rmax.txt', np.asarray([Rmax*1e-6]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'Nmax.txt', np.asarray([nuc_Nmax]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'UC.txt', np.asarray([underCoolingRate]), fmt='%1.4e',delimiter='\n')
    
    np.savetxt(mac_folder+'NG.txt', np.asarray([NG]), fmt='%d',delimiter='\n')
    np.savetxt(mac_folder+'NN.txt', np.asarray([NN]), fmt='%d',delimiter='\n')
    np.savetxt(mac_folder+'z0.txt', np.asarray([z0]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'r0.txt', np.asarray([r0]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'top.txt', np.asarray([top]), fmt='%1.4e',delimiter='\n')
    
    hf = h5py.File(mac_folder+'Temp.h5', 'w')
    hf.create_dataset('Temp', data=T)
    hf.close()
    
    cmd = "./phase_field " + args.meltpool + ".py"
    
    cmd = cmd + " -s " + str(args.seed) + " -b " + args.boundary + " -o " + args.outfile_folder
    
    if args.liquid:
        cmd = cmd + " -n 1"
    elif args.nucl:
        cmd = cmd + " -n 2"
    else:
        cmd = cmd
        
    if args.mpi>1:
        cmd = "ibrun -n " + str(args.mpi) + " " + cmd
        
    if args.line:
        cmd = cmd + " -l 1"
    
    print(cmd)
    
    os.system(cmd)  
