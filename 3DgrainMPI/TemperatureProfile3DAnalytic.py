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
        
        return -self.G*self.dist2Interface(profile, x, y, z, z0, r0) - self.U*t*1e6
    
    def dist2Interface(self, profile, x, y, z, z0=0, r0=0):
        
        if profile == 'uniform':
            return -10
        
        if profile == 'line':
            return self.lineProfile(x, y, z, z0)
        
        if profile == 'cylinder':
            return self.cylinderProfile(x, y, z, z0, r0, [self.ly/2, self.lz])
        
        if profile == 'sphere4':
            return self.sphereProfile(x, y, z, z0, r0, [self.lx, self.ly/2, self.lz])
        
        if profile == 'sphere8':
            return self.sphereProfile(x, y, z, z0, r0, [self.lx, self.ly, self.lz])

  

    def lineProfile(self, x, y, z, z0):
        

        return z0 - z
    
    
    """ y-z cross-section """
    def cylinderProfile(self, x, y, z, z0, r0, centerline):
        
        yc, zc = centerline
        dist = np.sqrt((y - yc)**2 + (z - z0 - zc)**2)

        return dist - r0
    
    
    def sphereProfile(self, x, y, z, z0, r0, center):
        
        xc, yc, zc = center
        dist = np.sqrt((x - xc)**2 + (y - yc)**2 + (z + z0 - zc)**2)
  
        return dist - r0





