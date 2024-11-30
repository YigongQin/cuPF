#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:11:06 2023

@author: yigongqin
"""

import numpy as np
from math import pi
from scipy.interpolate import griddata

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
    
    def dist2Interface(self, profile, x, y, z, z0=0, r0=0, angle = 0, min_angle = 0):
        
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
            
        if profile == 'paraboloid':
            return self.paraboloidProfile(x, y, z, z0, r0, [self.ly/2, self.lz + z0], angle, min_angle)

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
        lm = (r0-z0)/np.tan(angle)
        x_start = 0
        x_len_on_cone = x - x_start

        r0_x = z0 + (r0-z0)*x_len_on_cone/lm
        r0_x = np.clip(r0_x, a_min=None, a_max=r0)
        
        dist = np.sqrt((y - yc)**2 + (z - zc)**2)
       
        
        return dist - r0_x


    def paraboloidProfile(self, x, y, z, z0, r0, centerline, angle, min_angle):        
 
        Lx, Ly, Lz = self.lx, self.ly, self.lz

        dx = 0.5

        x1d = np.linspace(0, Lx, int(Lx/dx)+1)
        y1d = np.linspace(0, Ly, int(Ly/dx)+1)
        z1d = np.linspace(0, Lz, int(Lz/dx)+1)    
        xx, yy, zz = np.meshgrid(x1d, y1d, z1d, indexing='ij')

        
        rr = query_r(xx, z0, r0, angle, min_angle)
        dist = np.sqrt((zz-Lz-z0)**2 + (yy-Ly/2)**2)
        diff = np.absolute(dist-rr)
        xs, ys = np.meshgrid(x1d, y1d, indexing='ij') 
        zs = z1d[np.argmin(diff, axis=-1)]
        
        xs, ys, zs = xs.flatten(), ys.flatten(), zs.flatten()
        rs = query_r(xs, z0, r0, angle, min_angle)
        x_on = xs[z0**2 + (ys-Ly/2)**2 <= rs**2]
        y_on = ys[z0**2 + (ys-Ly/2)**2 <= rs**2]
        z_on = zs[z0**2 + (ys-Ly/2)**2 <= rs**2]

        x_out = xs[z0**2 + (ys-Ly/2)**2 > rs**2]
        y_out = ys[z0**2 + (ys-Ly/2)**2 > rs**2]
        z_out = zs[z0**2 + (ys-Ly/2)**2 > rs**2]
        rs_out = query_r(x_out, z0, r0, angle, min_angle)

        

        x_sam, y_sam, z_sam = x_on.copy(), y_on.copy(), z_on.copy()

        values = 0*x_sam

        x_sam, y_sam, z_sam = np.concatenate((x_sam, x_out)), np.concatenate((y_sam, y_out)), np.concatenate((z_sam, z_out))
        values = -np.concatenate((values, np.absolute(y_out-Ly/2) - np.sqrt(rs_out**2-z0**2) )) #

        xs, ys, zs = x_sam.copy(), y_sam.copy(), z_sam.copy()
        xb, yb, zb, values_b = x_sam.copy(), y_sam.copy(), z_sam.copy(), values.copy()

        
        for k in range(int(Lz/dx)):
            rs = query_r(xs, z0, r0, angle, min_angle)
            sin_beta = (ys - Ly/2)/rs
            tan_gamma = query_slope(xs, z0, r0, angle, min_angle)
            sin_gamma = tan_gamma/np.sqrt(1+tan_gamma**2)
            cos_gamma = 1/np.sqrt(1+tan_gamma**2)
            cos_beta = np.sqrt(1-sin_beta**2)
            normal = np.array([sin_gamma, -cos_gamma*sin_beta, cos_gamma*cos_beta])

            xs, ys, zs = xs + normal[0]*dx, ys + normal[1]*dx, zs + normal[2]*dx
            x_sam, y_sam, z_sam = np.concatenate((x_sam, xs.flatten())), np.concatenate((y_sam, ys.flatten())), np.concatenate((z_sam, zs.flatten()))

            values = np.concatenate((values, values_b+(k+1)*dx))


        for k in range(int(Lz/dx)):
            rs = query_r(xb, z0, r0, angle, min_angle)
            sin_beta = (yb - Ly/2)/rs
            tan_gamma = query_slope(xb, z0, r0, angle, min_angle)
            sin_gamma = tan_gamma/np.sqrt(1+tan_gamma**2)
            cos_gamma = 1/np.sqrt(1+tan_gamma**2)
            cos_beta = np.sqrt(1-sin_beta**2)
            normal = np.array([sin_gamma, -cos_gamma*sin_beta, cos_gamma*cos_beta])
            xb, yb, zb = xb - normal[0]*dx, yb - normal[1]*dx, zb - normal[2]*dx
            x_sam, y_sam, z_sam = np.concatenate((x_sam, xb.flatten())), np.concatenate((y_sam, yb.flatten())), np.concatenate((z_sam, zb.flatten()))

            values = np.concatenate((values, values_b-(k+1)*dx))



        y_sam = y_sam[x_sam<=Lx]
        z_sam = z_sam[x_sam<=Lx]
        values = values[x_sam<=Lx]
        x_sam = x_sam[x_sam<=Lx]

        x_sam = x_sam[z_sam<=Lz]
        y_sam = y_sam[z_sam<=Lz]
        values = values[z_sam<=Lz]
        z_sam = z_sam[z_sam<=Lz]

        y_sam = y_sam[x_sam>=0]
        z_sam = z_sam[x_sam>=0]
        values = values[x_sam>=0]
        x_sam = x_sam[x_sam>=0]

        x_sam = x_sam[z_sam>=0]
        y_sam = y_sam[z_sam>=0]
        values = values[z_sam>=0]
        z_sam = z_sam[z_sam>=0]

        distance = griddata((x_sam, y_sam, z_sam), values, (x, y, z), method='nearest')       
        
        return -distance


def query_r(x, z0, r0, angle, min_angle):
    k1 = np.tan(angle)
    k2 = np.tan(min_angle)
    lm = 2*(r0-z0)/(k1+k2)
    clip_x = np.clip(x, a_min=None, a_max=lm)

   # r0_x = z0 + k1*clip_x + (k2-k1)/(2*lm)*clip_x**2
   # r = 2+0.5*x 
   # return np.clip(r, a_min=2, a_max=20)
    return z0 + k1*clip_x + (k2-k1)/(2*lm)*clip_x**2

def query_slope(x, z0, r0, angle, min_angle):
    
   # slope = 0.5
    k1 = np.tan(angle)
    k2 = np.tan(min_angle)
    lm = 2*(r0-z0)/(k1+k2)
    clip_x = np.clip(x, a_min=None, a_max=lm)

    return k1*clip_x + (k2-k1)/(lm)*clip_x

