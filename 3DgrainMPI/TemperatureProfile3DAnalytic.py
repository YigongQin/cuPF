#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:11:06 2023

@author: yigongqin
"""

import numpy as np


class ThermalProfile:
    def __init__(self, domainSize, thermal):
        self.lx, self.ly, self.lz = domainSize
        self.G, self.R, self.U = thermal


    def pointwiseTempConstGR(self, profile, x, y, z, t, z0=0, r0=0):
        
        return self.G*self.dist2Interface(self, profile, x, y, z, z0, r0) - self.U*t
    
    def dist2Interface(self, profile, x, y, z, z0=0, r0=0):
        
        if profile == 'uniform':
            return 0
        
        if profile == 'line':
            return self.lineProfile(x, y, z, 2)
        
        if profile == 'cylinder':
            return self.cylinderProfile(x, y, z, r0, [self.lx/2, self.lz])
        
        if profile == 'sphere4':
            return self.sphereProfile(x, y, z, r0, [self.lx/2, self.ly, self.lz])
        
        if profile == 'sphere8':
            return self.sphereProfile(x, y, z, r0, [self.lx, self.ly, self.lz])

  

    def lineProfile(x, y, z,  z0):
        

        return z - z0
    
    
    """ x-z cross-section """
    def cylinderProfile(x, y, z, r0, centerline):
        
        x0, z0 = centerline
        dist = np.sqrt((x - x0)**2 + (z - z0)**2)

        return r0 - dist
    
    
    def sphereProfile(x, y, z, r0, center):
        
        x0, y0, z0 = center
        dist = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
  
        return r0 - dist





