import numpy as np
import h5py
import sys, os
from graph_datastruct import graph
from math import pi
import argparse
from TemperatureProfile3DAnalytic import ThermalProfile
from matplotlib import pyplot as plt
import glob

def stack_grid(outfile_folder, input_folder, direction, current_layer, z_stack = 20, y_stack=40):

    f_cur = h5py.File(outfile_folder + '/Powder_seed'+str(current_layer)+'.h5', 'r')
    cur_alpha = np.asarray(f_cur['alpha'])

    x = np.asarray(f_cur['x_coordinates'])
    y = np.asarray(f_cur['y_coordinates'])
    z = np.asarray(f_cur['z_coordinates'])
    fnx, fny, fnz = len(x), len(y), len(z)
    cur_alpha = cur_alpha.reshape((fnx, fny, fnz), order='F')

    f_cur.close()

    if current_layer == 1:
        f_prev = h5py.File(outfile_folder + '/Powder_seed0.h5', 'r')
    else:
        file = glob.glob(outfile_folder + '*seed'+str(current_layer-1)+'*Rmax*'+'.h5')[0]
        f_prev = h5py.File(file, 'r')

    prev_alpha = np.asarray(f_prev['alpha'])
    f_prev.close()


    f = h5py.File(input_folder+'/alpha3D.h5', 'w')
    
    f.create_dataset('x_coordinates', data=x)


    if direction == 'z':
        z_dim = 2*fnz-3
        print('alpha dim:', fnx, fny, z_dim)
        alpha_save = np.zeros((fnx, fny, z_dim), order='F', dtype=np.int32)
        if current_layer == 1:
            prev_alpha = prev_alpha.reshape((fnx, fny, fnz), order='F')
            alpha_save[:, :, :fnz-1] = prev_alpha[:, :, :fnz-1]
            alpha_save[:, :, fnz-1:] = cur_alpha[:, :, 2:]
        else:
            prev_alpha = prev_alpha.reshape((fnx, fny, z_dim), order='F')
            alpha_save[:, :, :fnz-1] = prev_alpha[:, :, fnz-2:]
            alpha_save[:, :, fnz-1:] = cur_alpha[:, :, 2:]

        f.create_dataset('alpha', data=alpha_save.reshape(fnx*fny*z_dim, order='F'))

    elif direction == 'y':
        y_dim = 2*fny-3
        print('alpha dim:', fnx, y_dim, fnz)
        alpha_save = np.zeros((fnx, y_dim, fnz), order='F', dtype=np.int32)

        if current_layer == 1:
            prev_alpha = prev_alpha.reshape((fnx, fny, fnz), order='F')
            alpha_save[:, :fny-1, :] = prev_alpha[:, :fny-1, :]
            alpha_save[:, fny-1:, :] = cur_alpha[:, 2:, :]
        else:
            prev_alpha = prev_alpha.reshape((fnx, y_dim, fnz), order='F')
            prev_alpha = prev_alpha[:, ::-1, :]
            alpha_save[:, :fny-1, :] = prev_alpha[:, fny-2:, :]
            alpha_save[:, fny-1:, :] = cur_alpha[:, 2:, :]

        f.create_dataset('alpha', data=alpha_save.reshape(fnx*y_dim*fnz, order='F'))



    else:
        print('wrong direction')
        return

    f.close()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Generate thermal input for PF")
    parser.add_argument("--outfile_folder", type=str, default = '/scratch1/07428/ygqin/graph/cone_test/')
    parser.add_argument("--mode", type=str, default = 'check')
    parser.add_argument("--seed", type=int, default = 0)
    parser.add_argument("--save3Ddata", type=int, default = 0)
    parser.add_argument("--meltpool", type=str, default = 'paraboloid')
    parser.add_argument("--boundary", type=str, default = '000')
    parser.add_argument("--mpi", type=int, default = 1)
    parser.add_argument("--G", type=float, default = 8)
    parser.add_argument("--Rmax", type=float, default = 1.5)
    parser.add_argument("--sin_gamma", type=float, default = 0.7)

    parser.add_argument("--nucleation", dest='nucl', action='store_true')
    parser.set_defaults(nucl=False)

    parser.add_argument("--liquidStart", dest='liquid', action='store_true')
    parser.set_defaults(liquid=False)

    parser.add_argument("--lineConfig", dest='line', action='store_true')
    parser.set_defaults(line=False)

    parser.add_argument("--save_phi", dest='save_phi', action='store_true')
    parser.set_defaults(save_phi=False)

    parser.add_argument("--save_T", dest='save_T', action='store_true')
    parser.set_defaults(save_T=True)    

    parser.add_argument("--powder", dest='powder', action='store_true')
    parser.set_defaults(powder=False)

    parser.add_argument("--scan", dest='scan', action='store_true')
    parser.set_defaults(scan=False)

    parser.add_argument("--layers", type=int, default = 1)
    parser.add_argument("--build_direction", type=str, default = 'z')

    args = parser.parse_args()     
    

    seed = args.seed
    
    if args.meltpool == 'cylinder':
        from cylinder import *
    
    if args.meltpool == 'line':
        from line import *    

    if args.meltpool == 'lineTemporal':
        from lineTemporal import *

    if args.meltpool == 'cone_long':
        from cone_long import *

    if args.meltpool == 'cone':
        from cone import *

    if args.meltpool == 'paraboloid':
        from paraboloid import *


    
    if args.meltpool == 'cone':

        underCoolingRate = G*Rmax/1e6
        sin_gamma = Rmax/V
        cos_gamma = np.sqrt(1-sin_gamma**2)
        r0 = 76*sin_gamma*cos_gamma
        assert r0>z0 

    if args.meltpool == 'paraboloid':
        underCoolingRate = G*Rmax/1e6 


    bc = 'periodic' if (args.boundary)[:2] == '11' else 'noflux'

    # create nucleation pool
    num_nucleatioon_theta = 1000
    
    # sample orientations
    np.random.seed(args.seed)
    ux = np.random.randn(num_nucleatioon_theta)
    uy = np.random.randn(num_nucleatioon_theta)
    uz = np.random.randn(num_nucleatioon_theta)
    
    u = np.sqrt(ux**2+uy**2+uz**2)
    ux = ux/u
    uy = uy/u
    uz = uz/u
    
    theta_x = np.zeros(1 + num_nucleatioon_theta)
    theta_z = np.zeros(1 + num_nucleatioon_theta)
    theta_x[1:] = np.arctan2(uy, ux)%(pi)
    theta_z[1:] = np.arctan2(np.sqrt(ux**2+uy**2), uz)%(pi)

    NG = num_nucleatioon_theta
    NN = 2*num_nucleatioon_theta
    theta = np.hstack([0, theta_x[1:], theta_z[1:]])

    print('no. nodes', NN, 'no. regions', NG)
    
    x = np.linspace(0-BC,Lx+BC,nx)
    y = np.linspace(0-BC,Ly+BC,ny)
    z = np.linspace(0-BC,Lz+BC,nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    therm = ThermalProfile([Lx, Ly, Lz], [G, Rmax, underCoolingRate], seed=seed)
    
    angle = 0
    min_angle = 0

    t_end = (track+40)/V

    t = np.linspace(0, t_end, nt)
    T = np.zeros(nx*ny*nz*nt)

    if args.meltpool == 'cone' or args.meltpool == 'paraboloid':
        angle = np.arcsin(Rmax/V)   
    else:
        angle = 0

    if args.meltpool == 'paraboloid':
        min_angle = np.arcsin(Rmin/V)
    else:
        min_angle = 0

    psi3d = therm.dist2Interface(args.meltpool, xx, yy, zz, z0=z0, r0=r0, angle = angle, min_angle=min_angle)  
    psi = psi3d.reshape(nx*ny*nz, order='F')

        
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
   # np.savetxt(mac_folder+'alpha.txt', alpha, fmt='%d',delimiter='\n')
    np.savetxt(mac_folder+'theta.txt', theta, fmt='%1.4e',delimiter='\n')
    
    np.savetxt(mac_folder+'G.txt', np.asarray([G]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'Rmax.txt', np.asarray([Rmax*1e-6]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'V.txt', np.asarray([V*1e-6]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'Nmax.txt', np.asarray([nuc_Nmax]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'UC.txt', np.asarray([underCoolingRate]), fmt='%1.4e',delimiter='\n')
    
    np.savetxt(mac_folder+'NG.txt', np.asarray([NG]), fmt='%d',delimiter='\n')
    np.savetxt(mac_folder+'NN.txt', np.asarray([NN]), fmt='%d',delimiter='\n')
    np.savetxt(mac_folder+'z0.txt', np.asarray([z0]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'r0.txt', np.asarray([r0]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'top.txt', np.asarray([top]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'angle.txt', np.asarray([angle]), fmt='%1.4e',delimiter='\n')
    np.savetxt(mac_folder+'min_angle.txt', np.asarray([min_angle]), fmt='%1.4e',delimiter='\n')
    if args.meltpool == 'paraboloid':
        np.savetxt(mac_folder+'mp_len.txt', np.asarray([therm.mp_len]), fmt='%1.4e',delimiter='\n')


    if args.save_phi:
        from tvtk.api import tvtk, write_data
        grid = tvtk.ImageData(spacing=(0.5,0.5,0.5), origin=(0, 0, 0), 
                              dimensions=psi3d.shape)
    
        grid.point_data.scalars = np.tanh(psi3d).ravel(order='F')
  
        write_data(grid, 'phi.vtk') 


    hf = h5py.File(mac_folder+'Temp.h5', 'w')
    hf.create_dataset('Temp', data=T)
    hf.close()


    cmd = "./phase_field " + args.meltpool + ".py" + " -b " + args.boundary + " -o " + args.outfile_folder

    if args.mpi>1:
        cmd = "ibrun -n " + str(args.mpi) + " " + cmd
        
    if args.line:
        cmd = cmd + " -l 1"

    if args.powder:
        for layer in range(args.layers+1):
            runcmd = cmd + " -s " + str(layer) +  " -n 1"
            print(runcmd)
            os.system(runcmd) 

    if args.scan:
        z_stack = 0.5*Lz
        y_stack = 0.5*Ly
        for layer in range(args.layers):
            stack_grid(args.outfile_folder, mac_folder, args.build_direction, layer+1, z_stack, y_stack)
            runcmd = cmd + " -s " + str(layer+1) + " -p 1"
            print(runcmd)
            os.system(runcmd)



