#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:11:06 2023

@author: yigongqin
"""

import numpy as np
from math import pi
from scipy.interpolate import griddata, interp1d

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
    
    def dist2Interface(self, profile, x, y, z, z0=0, r0=0, x0 = 0, angle = 0, min_angle = 0, order = 2, includeG = False):
        
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
            return self.paraboloidProfile(x, y, z, z0, r0, x0, [self.ly/2, self.lz + z0], angle, min_angle, order, includeG)

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


    def paraboloidProfile(self, x, y, z, z0, r0, x0, centerline, angle, min_angle, order, includeG):

        if order == 100:
            k1 = k2 = 0
            x_macro = np.load(self.macro_dir + 'x_coords.npy')
           # print(x_macro)
            self.mp_len = x_macro[-1] - x_macro[0]
        else:
            k1, k2 = np.tan(angle), np.tan(min_angle)
            self.mp_len = order*(r0-z0)/(k2+(order-1)*k1)        
        # self.mp_len = 2*(r0-z0)/(np.tan(angle)+np.tan(min_angle))
        print('z0, r0, k1, k2, x0: ', z0, r0, k1, k2, x0)
        print('profile order: ', order)
        print('angle, min angle: ', angle, min_angle)
        print('mp len: ', self.mp_len)
        Lx, Ly, Lz = self.lx, self.ly, self.lz

        dx = 0.5
        
        window_len = self.mp_len + x0
        x1d = np.linspace(0, window_len, int(window_len/dx)+1)
        y1d = np.linspace(0, Ly, int(Ly/dx)+1)
        z1d = np.linspace(0, Lz, int(Lz/dx)+1)    
        xx, yy, zz = np.meshgrid(x1d, y1d, z1d, indexing='ij')

        
        rr = query_r(xx, z0, r0, k1, k2, x0, order, self.macro_dir)
        dist = np.sqrt((zz-Lz-z0)**2 + (yy-Ly/2)**2)
        diff = np.absolute(dist-rr)
        xs, ys = np.meshgrid(x1d, y1d, indexing='ij') 
        zs = z1d[np.argmin(diff, axis=-1)]
        
        xs, ys, zs = xs.flatten(), ys.flatten(), zs.flatten()
        rs = query_r(xs, z0, r0, k1, k2, x0, order, self.macro_dir)
        x_on = xs[z0**2 + (ys-Ly/2)**2 <= rs**2]
        y_on = ys[z0**2 + (ys-Ly/2)**2 <= rs**2]
        z_on = zs[z0**2 + (ys-Ly/2)**2 <= rs**2]

        

        x_sam, y_sam, z_sam = x_on.copy(), y_on.copy(), z_on.copy()

        values = 0*x_sam

       # x_sam, y_sam, z_sam = np.concatenate((x_sam, x_out)), np.concatenate((y_sam, y_out)), np.concatenate((z_sam, z_out))
       # values = -np.concatenate((values, np.absolute(y_out-Ly/2) - np.sqrt(rs_out**2-z0**2) )) #

        xs, ys, zs = x_sam.copy(), y_sam.copy(), z_sam.copy()
        xb, yb, zb = x_sam.copy(), y_sam.copy(), z_sam.copy()

        tan_gamma = query_slope(xs, z0, r0, k1, k2, x0, order, self.macro_dir)
        sin_gamma = tan_gamma/np.sqrt(1+tan_gamma**2)
        G = query_G(xs, z0, r0, k1, k2, x0, order, self.macro_dir, includeG)
        print('G values on inteface:', G)

        values_b = values.copy()
        cur_val = values_b.copy()
       # print(xs)
        for k in range(int(self.mp_len/dx)):

            
            f = interp1d(xs, cur_val, kind='linear', fill_value=(cur_val[0], cur_val[-1]), bounds_error=False)

            xs = xs + dx
            x_sam, y_sam, z_sam = np.concatenate((x_sam, xs.flatten())), np.concatenate((y_sam, ys.flatten())), np.concatenate((z_sam, zs.flatten()))

            cur_val = f(xs - dx*sin_gamma**2) + dx*sin_gamma*G
           # print(cur_val)

            values = np.concatenate((values, cur_val))

        cur_val = values_b.copy()
        for k in range(int(x0/dx)):

            f = interp1d(xb, cur_val, kind='linear', fill_value=(cur_val[0], cur_val[-1]), bounds_error=False)

            xb = xb - dx
            x_sam, y_sam, z_sam = np.concatenate((x_sam, xb.flatten())), np.concatenate((y_sam, yb.flatten())), np.concatenate((z_sam, zb.flatten()))

            cur_val = f(xb + dx*sin_gamma**2) - dx*sin_gamma*G

            values = np.concatenate((values, cur_val))

        print("min and max values:  ", np.min(values), np.max(values))


        y_sam = y_sam[x_sam<=window_len]
        z_sam = z_sam[x_sam<=window_len]
        values = values[x_sam<=window_len]
        x_sam = x_sam[x_sam<=window_len]

        y_sam = y_sam[x_sam>=0]
        z_sam = z_sam[x_sam>=0]
        values = values[x_sam>=0]
        x_sam = x_sam[x_sam>=0]

        distance = 0*x
        window_len_n = int(window_len/(x[1,0,0]-x[0,0,0]))
        print("window_len_n", window_len_n)

        distance[:window_len_n] = griddata((x_sam, y_sam, z_sam), values, (x[:window_len_n], y[:window_len_n], z[:window_len_n]), method='nearest')
        dist = np.sqrt((y - Ly/2)**2 + (z - Lz - z0)**2) 
        distance[window_len_n:] = r0 - dist[window_len_n:]

        distance[dist>r0] = r0 - dist[dist>r0] 

        return -distance


def query_r(x, z0, r0, k1, k2, x0, order, macro_dir):

    if order == 100:
        x_macro = np.load(macro_dir + 'x_coords.npy')
        r_macro = np.load(macro_dir + 'radius_vals.npy')
        f = interp1d(x_macro, r_macro, kind='linear', fill_value=(r_macro[0], r_macro[-1]), bounds_error=False)
        clip_x = np.clip(x-x0, a_min=-0.1, a_max=x_macro[-1]-x_macro[0])
        return f(clip_x)

    lm = order*(r0-z0)/(k2+(order-1)*k1)
    clip_x = np.clip(x-x0, a_min=-0.1, a_max=lm)

    return z0 + k1*clip_x + (k2-k1)/(order*lm**(order-1))*clip_x**order
   # return z0 + k1*clip_x + (k2-k1)/(2*lm)*clip_x**2

def query_slope(x, z0, r0, k1, k2, x0, order, macro_dir):

    if order == 100:
        x_macro = np.load(macro_dir + 'x_coords.npy')
        r_macro = np.load(macro_dir + 'radius_vals.npy')
        df_forward = (r_macro[1:] - r_macro[:-1]) / (x_macro[1:] - x_macro[:-1])
        df_central = (df_forward[:-1] + df_forward[1:]) / 2
        slope_macro = np.concatenate(([df_forward[0]], df_central, [df_forward[-1]]))
        f = interp1d(x_macro, slope_macro, kind='linear', fill_value=(slope_macro[0], slope_macro[-1]), bounds_error=False)
        clip_x = np.clip(x-x0, a_min=-0.1, a_max=x_macro[-1]-x_macro[0])
        return f(clip_x)

    lm = order*(r0-z0)/(k2+(order-1)*k1)
    clip_x = np.clip(x-x0, a_min=-0.1, a_max=lm)

    return k1 + (k2-k1)/(lm**(order-1))*clip_x**(order-1)
   # return k1 + (k2-k1)/(lm)*clip_x

def query_G(x, z0, r0, k1, k2, x0, order, macro_dir, includeG):
    if not includeG or order != 100:
        return 1
    x_macro = np.load(macro_dir + 'x_coords.npy')
    G_macro = np.load(macro_dir + 'G_vals.npy')
    f = interp1d(x_macro, G_macro, kind='linear', fill_value=(G_macro[0], G_macro[-1]), bounds_error=False)
    clip_x = np.clip(x-x0, a_min=-0.1, a_max=x_macro[-1]-x_macro[0])
    return f(clip_x)