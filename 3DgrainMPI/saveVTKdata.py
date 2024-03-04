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
       
        for rank in range(self.gpus):
            
            self.data_file = (glob.glob(rawdat_dir + '/*seed'+str(self.seed)+ '*rank'+str(rank) + '_' + '*time'+str(self.time) + '*.h5'))[0]
            f = h5py.File(self.data_file, 'r')
            
            self.x = np.asarray(f['x_coordinates'])
            self.y = np.asarray(f['y_coordinates'])
            self.z = np.asarray(f['z_coordinates']) 
        
            fnx, fny, fnz = len(self.x), len(self.y), len(self.z)
            print('grid ', fnx, fny, fnz)
            
            alpha_pde = np.asarray(f['alpha']).reshape((fnx, fny, fnz),order='F')    
         
         
            x_list.append(self.x)
            alpha_list.append(alpha_pde)


            if rank == 0:
                self.angles = np.asarray(f['angles']) 
                num_theta = len(self.angles)//2
                self.theta_z = np.zeros(1 + num_theta)
                self.theta_z[1:] = self.angles[num_theta+1:]
                
                dx = self.x[1] - self.x[0]
                G = re.search('G(\d+\.\d+)', self.data_file).group(1)
                UC = re.search('UC(\d+\.\d+)', self.data_file).group(1)
              #  data_frames = int(re.search('frames(\d+)', self.data_file).group(1))+1
                self.physical_params = {'G':G, 'UC':UC}
                print(self.physical_params)

 
        self.x = np.concatenate(x_list)
        self.alpha_pde = np.concatenate(alpha_list, axis=0)
        print('x, y, z', self.x.shape, self.y.shape, self.z.shape)
        print('alpha_pde', self.alpha_pde.shape)
 
    
        top_z = int(np.round(self.height/dx))
        
        if self.height == -1:
            self.alpha_pde = self.alpha_pde[1:-1, 1:-1, 1:-1]
        
        else:
            self.alpha_pde = self.alpha_pde[1:-1, 1:-1, 1:top_z]       
        

        
        print(len(np.unique(self.alpha_pde)), np.unique(self.alpha_pde))
      #  self.alpha_pde[self.alpha_pde == 0] = np.nan
        angle_pde = (self.alpha_pde<=num_theta)*self.alpha_pde + (self.alpha_pde>num_theta)*(self.alpha_pde%num_theta+1) 
        self.alpha_pde = self.theta_z[angle_pde]/pi*180
        
        origin = (self.x[1], self.y[1], self.z[1]) 

    
        self.alpha_pde = self.alpha_pde[::self.subsample,::self.subsample,::self.subsample]
        dx = dx*self.subsample
        grid = tvtk.ImageData(spacing=(dx, dx, dx), origin=origin, 
                              dimensions=self.alpha_pde.shape)
        
        grid.point_data.scalars = self.alpha_pde.ravel(order='F')
        grid.point_data.scalars.name = 'theta_z'
        
        
        
        self.dataname = rawdat_dir + '/seed'+str(self.seed) + '_time'+ str(self.time) +'.vtk'
                   #rawdat_dir + 'seed'+str(self.seed)+'_G'+str('%2.2f'%self.physical_params['G'])\
                   #+'_R'+str('%2.2f'%self.physical_params['R'])+'.vtk'
        write_data(grid, self.dataname)
        



if __name__ == '__main__':


    parser = argparse.ArgumentParser("create 3D grain plots with paraview")

    parser.add_argument("--rawdat_dir", type=str, default = './')
    parser.add_argument("--pvpython_dir", type=str, default = '')
    parser.add_argument("--mode", type=str, default='truth')
    
    parser.add_argument("--seed", type=int, default = 0)
    parser.add_argument("--gpus", type=int, default = 0)
    parser.add_argument("--time", type=int, default = 0)
    parser.add_argument("--lxd", type=int, default = 40)
    parser.add_argument("--height", type=int, default = -1)
    parser.add_argument("--subsample", type=int, default = 2)

    args = parser.parse_args()
        
   # Gv = grain_visual(lxd = 20, seed = args.seed, height=20)   
    Gv = grain_visual(gpus = args.gpus, lxd=args.lxd, seed=args.seed, height=args.height, time=args.time, subsample=args.subsample)  
    if args.mode == 'truth':
        
        Gv.load(rawdat_dir=args.rawdat_dir)  
        

