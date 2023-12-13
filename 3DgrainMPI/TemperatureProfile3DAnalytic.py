#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:11:06 2023

@author: yigongqin
"""

import numpy as np
from math import pi

class ThermalProfile:
    def __init__(self, domainSize, thermal, seed):
        self.lx, self.ly, self.lz = domainSize
        self.G, self.R, self.U = thermal
        self.seed = seed

    @staticmethod
    def RandGR(t, t_end, t_sampling_freq):
        
        G_freq = np.arange(1,t_sampling_freq+1)/t_end*pi/2
        G_coeff = np.random.rand(len(G_freq))
        G_phase = np.random.rand(len(G_freq))*2*pi
        print(G_freq, G_coeff, G_phase)

        R_freq = np.arange(1,t_sampling_freq+1)/t_end*pi/2
        R_coeff = np.random.rand(len(R_freq))
        R_phase = np.random.rand(len(R_freq))*2*pi

        G, R = np.zeros(len(t)), np.zeros(len(t))

        for i in range(t_sampling_freq):
           # print(G_coeff, G_freq[i], z, G_phase)
            G += G_coeff[i]*np.cos(G_freq[i]*t+G_phase[i])/(i+1)
            R += R_coeff[i]*np.sin(R_freq[i]*t+R_phase[i])/(i+1)

       # G = np.mean(np.expand_dims(G_coeff, axis=-1)*np.cos(np.outer(G_freq, z) + np.expand_dims(G_phase, axis=-1)) , axis=0)
       # R = np.mean(np.expand_dims(R_coeff, axis=-1)*np.sin(np.outer(R_freq, z) + np.expand_dims(R_phase, axis=-1)) , axis=0)    
        
        G = 0.5 + 9.5*(G-np.min(G))/(np.max(G)-np.min(G))
        R = 0.2 + 1.8*(R-np.min(R))/(np.max(R)-np.min(R))
        
        return G, R

    def pointwiseTempConstGR(self, profile, x, y, z, t, z0=0, r0=0, angle = 0):
        
        return -self.G*self.dist2Interface(profile, x, y, z, z0, r0, angle) - self.U*t*1e6
    
    def dist2Interface(self, profile, x, y, z, z0=0, r0=0, angle = 0):
        
        if profile == 'uniform':
            return -10
        
        if profile == 'line':
            return self.lineProfile(x, y, z, z0)
        
        if profile == 'cylinder':
            return self.cylinderProfile(x, y, z, r0, [self.ly/2, self.lz + z0])
        
        if profile == 'sphere4':
            return self.sphereProfile(x, y, z, r0, [self.lx, self.ly/2, self.lz + z0])
        
        if profile == 'sphere8':
            return self.sphereProfile(x, y, z,r0, [self.lx, self.ly, self.lz + z0])

        if profile == 'cone':
            return self.coneProfile(x, y, z, z0, r0, [self.ly/2, self.lz + z0], angle)
            


    def lineProfile(self, x, y, z, z0):        

        return z0 - z
    
    
    """ y-z cross-section """
    def cylinderProfile(self, x, y, z, r0, centerline):
        
        yc, zc = centerline
        dist = np.sqrt((y - yc)**2 + (z - zc)**2)

        return dist - r0
    
    
    def sphereProfile(self, x, y, z, r0, center):
        
        xc, yc, zc = center
        dist = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2)
  
        return dist - r0


    def coneProfile(self, x, y, z, z0, r0, centerline, angle):
        
        yc, zc = centerline
        lm = (r0-z0)/np.sin(angle)
        x_start = 0
        z_dist = zc - z
        z_tilt = z_dist/np.cos(angle)
        x_len_on_cone = x + (z_dist - z0)*np.tan(angle) - x_start
        
        if x_len_on_cone > lm:
            return (x_len_on_cone - lm)*np.cos(angle) - 0.4

        r0_x = z0 + (r0-z0)*x_len_on_cone/lm
        
        dist = np.sqrt((y-yc)**2 + z_tilt**2)
        
        return dist - r0_x






