#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:37:04 2023

@author: yigongqin
"""

import h5py, glob, re, os, argparse
from  math import pi
import numpy as np
from tvtk.api import tvtk, write_data




class grain_visual:
    def __init__(self, 
                 gpus: int = 1,
                 lxd: float = 40, 
                 seed: int = 1, 
                 frames: int = 1, 
                 height: int = -1,
                 time: int = 0,
                 subsample: int = 2,
                 physical_params = {}):   

        self.gpus = gpus
        self.lxd = lxd
        self.seed = seed
        self.height = height
        self.time = time
        self.subsample = subsample
        self.base_width = 2
        self.frames = frames # note that frames include the initial condition
        self.physical_params = physical_params
        

    def load(self, rawdat_dir: str = './'):
       
        x_list = []
        alpha_list = []
        save_file = None
        for rank in range(self.gpus):
            
            self.data_file = (glob.glob(rawdat_dir + '/*seed'+str(self.seed)+ '*rank'+str(rank) + '_' + '*time'+str(self.time) + '*.h5'))[0]
            f = h5py.File(self.data_file, 'r')
            
            self.x = np.asarray(f['x_coordinates'])
            self.y = np.asarray(f['y_coordinates'])
            self.z = np.asarray(f['z_coordinates']) 
        
            fnx, fny, fnz = len(self.x), len(self.y), len(self.z)
            print('grid ', fnx, fny, fnz)
            
            alpha_pde = np.asarray(f['alpha']).reshape((fnx, fny, fnz),order='F')    
         
            if rank>0:
                self.x = self.x[1:]
                alpha_pde = alpha_pde[1:,:,:]
            if rank<self.gpus-1:
                self.x = self.x[:-1]
                alpha_pde = alpha_pde[:-1,:,:]

            x_list.append(self.x)
            alpha_list.append(alpha_pde)


            if rank == 0:
                self.angles = np.asarray(f['angles']) 
                num_theta = len(self.angles)//2
                self.num_theta = num_theta
                self.theta_z = np.zeros(1 + num_theta)
                self.theta_z[1:] = self.angles[num_theta+1:]
                
                dx = self.x[1] - self.x[0]
                save_file = self.data_file
            f.close() 
        self.x = np.concatenate(x_list)
        self.alpha_pde = np.concatenate(alpha_list, axis=0)
        print('x, y, z', self.x.shape, self.y.shape, self.z.shape)
        print('alpha_pde', self.alpha_pde.shape)
 

        top_z = int(np.round(self.height/dx))
        
        if self.height == -1:
            self.alpha_pde = self.alpha_pde[1:-1, 1:-1, 1:-1]
        
        else:
            self.alpha_pde = self.alpha_pde[1:-1, 1:-1, 1:top_z]       
        
        self.alpha_pde = self.alpha_pde[::self.subsample,::self.subsample,::self.subsample]
        self.dx = dx*self.subsample

        f = h5py.File(rawdat_dir+'seed'+str(self.seed)+'_time'+str(self.time)+'.h5', 'w')
        f.create_dataset('alpha', data=self.alpha_pde.reshape(self.alpha_pde.shape[0]*self.alpha_pde.shape[1]*self.alpha_pde.shape[2], order='F'))
        f.create_dataset('subsample', data=self.subsample)
        f.close()


    def save_vtk(self, rawdat_dir):
        
        print('save vtk data') 
        
        #print(len(np.unique(self.alpha_pde)), np.unique(self.alpha_pde))
      #  self.alpha_pde[self.alpha_pde == 0] = np.nan
        angle_pde = (self.alpha_pde<=self.num_theta)*self.alpha_pde + (self.alpha_pde>self.num_theta)*(self.alpha_pde%self.num_theta+1) 
        self.alpha_pde = self.theta_z[angle_pde]/pi*180
        
        origin = (self.x[1], self.y[1], self.z[1]) 


        grid = tvtk.ImageData(spacing=(self.dx, self.dx, self.dx), origin=origin, 
                              dimensions=self.alpha_pde.shape)
        
        grid.point_data.scalars = self.alpha_pde.ravel(order='F')
        grid.point_data.scalars.name = 'theta_z'
        
        self.dataname = rawdat_dir + '/seed'+str(self.seed) + '_time'+ str(self.time) +'.vtk'
                   #rawdat_dir + 'seed'+str(self.seed)+'_G'+str('%2.2f'%self.physical_params['G'])\
                   #+'_R'+str('%2.2f'%self.physical_params['R'])+'.vtk'
        write_data(grid, self.dataname)

    def updateh5(self, rawdat_dir):

        data_file = (glob.glob(rawdat_dir + '/*seed'+str(self.seed)+ '*rank0.h5'))[0]
        f = h5py.File(data_file, 'r+')
        self.geometry = {'r0':float(np.array(f['r0'])), 'z0':float(np.array(f['z0'])), 'angle':float(np.array(f['angle']))}

        x = self.x*np.cos(self.geometry['angle'])
        dx = x[1] - x[0]
        y_max = 2*np.arccos(self.geometry['z0']/self.geometry['r0'])*self.geometry['r0']
        
        y_max_r = dx*int(y_max/dx)
        y = np.arange(-dx, y_max_r+2*dx, dx)
        
        x_in = x[1:-1]
        y_in = y[1:-1]
        
        surface_alpha = np.zeros((len(x_in), len(y_in)), dtype=int)
        print(dx, self.dx, surface_alpha.shape) 
        for i in range(surface_alpha.shape[0]):
            for j in range(surface_alpha.shape[1]):
                
                theta = dx*(j - (surface_alpha.shape[1]-1)/2 )/self.geometry['r0']
                y_len = self.y[-2]/2 + self.geometry['r0']*np.sin(theta)
                ai = i 
                aj = int( y_len/self.dx )
                z_len = self.z[-2] + self.geometry['z0'] - self.geometry['r0']*np.cos(theta)
                ak = int( z_len/self.dx )
                
                surface_alpha[i,j] = self.alpha_pde[ai, aj, ak]

        print(surface_alpha)
        
        
        if 'manifold' in f: del f['manifold']
        f['manifold'] = surface_alpha
        
        if 'x_manifold' in f: del f['x_manifold']
        f['x_manifold'] = x
        
        if 'y_manifold' in f: del f['y_manifold']
        f['y_manifold'] = y

        if 'x_coordinates' in f: del f['x_coordinates']
        f['x_coordinates'] = self.x

        f.close()
 

if __name__ == '__main__':


    parser = argparse.ArgumentParser("create 3D grain plots with paraview")

    parser.add_argument("--rawdat_dir", type=str, default = './')
    parser.add_argument("--pvpython_dir", type=str, default = '')
    parser.add_argument("--mode", type=str, default='plot')
    
    parser.add_argument("--seed", type=int, default = 0)
    parser.add_argument("--gpus", type=int, default = 0)
    parser.add_argument("--time", type=int, default = 0)
    parser.add_argument("--lxd", type=int, default = 40)
    parser.add_argument("--height", type=int, default = -1)
    parser.add_argument("--subsample", type=int, default = 2)

    args = parser.parse_args()
        
   # Gv = grain_visual(lxd = 20, seed = args.seed, height=20)   
    Gv = grain_visual(gpus = args.gpus, lxd=args.lxd, seed=args.seed, height=args.height, time=args.time, subsample=args.subsample) 
    Gv.load(rawdat_dir=args.rawdat_dir)

    if args.mode == 'plot':
        
        Gv.save_vtk(rawdat_dir=args.rawdat_dir)  
    if args.mode == 'manifold':
        Gv.updateh5(rawdat_dir=args.rawdat_dir)



        """
        lxd, lyd, lzd = self.x[-2], self.y[-2], self.z[-2]
        xx, yy, zz = np.meshgrid(self.x[1:-1:self.subsample], self.y[1:-1:self.subsample], self.z[1:-1:self.subsample], indexing='ij')

        center_height = lzd + self.geometry['z0']*np.cos(self.geometry['angle'])
        radius = self.geometry['r0']*np.cos(self.geometry['angle'])
        print('center_height, radius', center_height, radius)


        y_span = np.sqrt(radius**2 - (center_height - lzd)**2)
        y_half = int(np.round(lyd/2/self.dx))
        y_range = int( y_span/self.dx ) 
        self.alpha_pde = self.alpha_pde[:, y_half-y_range:y_half+y_range+1, :]
        yy =  yy[:, y_half-y_range:y_half+y_range+1, :]
        print('range of y', lyd/2 - y_span, lyd/2 + y_span)
        print('shape of alpha_pde, yy', self.alpha_pde.shape, yy.shape) 
        assert np.all(radius**2 - (yy - lyd/2)**2)>0
        surface_z = np.asarray((center_height-np.sqrt(radius**2 - (yy[:,:,0] - lyd/2)**2))/self.dx, dtype=int )
        print(surface_z)
        surface_alpha = 0*surface_z
        
        """
