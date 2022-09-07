#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:27:17 2022

@author: yigongqin
"""


import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from math import pi

def in_bound(x, y):
    eps = 1e-12
    if x>=-eps and x<=1+eps and y>=-eps and y<=1+eps:
        return True
    else:
        return False

def hexagonal_lattice(dx=0.05, noise=0.0001, BC='periodic'):
    # Assemble a hexagonal lattice
    rows, cols = int(1/dx)+1, int(1/dx)
    print('cols, rows of grains', cols, rows)
    points = []
    in_points = []
    randNoise = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2)*noise, size=rows*cols*5)
    count = 0
    for row in range(rows*2):
        for col in range(cols):
            count+=1
            x = ( (col + (0.5 * (row % 2)))*np.sqrt(3) )*dx
            y = row*0.5 *dx
         
            x += randNoise[count,0]
            y += randNoise[count,1]
            
            if in_bound(x, y):
              in_points.append([x,y])
              points.append([x,y])
              if BC == 'noflux':
                  points.append([-x,y])
                  points.append([x,-y])
                  points.append([2-x,y])
                  points.append([x,2-y])
              if BC == 'periodic':
                  points.append([x+1,y])
                  points.append([x-1,y])
                  points.append([x,y+1])
                  points.append([x,y-1])                  
                  

    return points, in_points




        
class graph:
    def __init__(self, size, density = 0.05, noise =0.00005, seed = 1, BC = 'periodic', randInit = True):

        self.imagesize = size
        self.vertices = [] ## vertices coordinates
        self.edges = []  ## index linkage
        self.edge_colors = []  ## defined by region orientation 
        self.regions = [] ## index group
        self.region_colors = []  ##color
       # self.region_coors = [] ## region corner coordinates
        self.density = density
        self.noise = noise
        self.BC = BC
        self.alpha_field = np.zeros((size[0], size[1]), dtype=int)

        
        
        if randInit:
            np.random.seed(seed)
            self.inbound_random_voronoi()    
            self.plot_polygons(pic_size=self.imagesize)
            self.num_pf = len(self.regions)
            self.color_choices = np.zeros(2*self.num_pf+1)
            self.color_choices[1:] = -pi/2*np.random.random_sample(2*self.num_pf)
        
    def inbound_random_voronoi(self):

        mirrored_seeds, seeds = hexagonal_lattice(dx=self.density, noise = self.noise, BC = self.BC)
        vor = Voronoi(mirrored_seeds)     
    
        regions = []
       # reordered_regions = []
       # vertices = []
        vert_map = {}
        vert_count = 0
        alpha, beta = 0,0 
       # edges = []
        
        for region in vor.regions:
            flag = True
            for index in region:
                if index == -1:
                    flag = False
                    break
                else:
                    x = vor.vertices[index, 0]
                    y = vor.vertices[index, 1]
                    if not in_bound(x,y):
                        flag = False
                        break
            if region != [] and flag:
                
                '''
                valid region propertities
                '''
                
                regions.append(region)
                reordered_region = []
                alpha = alpha+1 #random.randint(1, self.num_pf)
                beta = beta+1 #random.randint(1, self.num_pf)
                self.region_colors.append((alpha, beta))
                
                for index in region:
                    point = tuple(vor.vertices[index])
                    if point not in vert_map:
                        self.vertices.append(point)
                        vert_map[point] = vert_count
                        reordered_region.append(vert_count)
                        vert_count += 1
                    else:
                        reordered_region.append(vert_map[point])
                for i in range(len(reordered_region)):
                    cur = vor.vertices[region[i]]
                    nxt = vor.vertices[region[i+1]] if i<len(region)-1 else vor.vertices[region[0]]
                    self.edges.append([vert_map[tuple(cur)], vert_map[tuple(nxt)]])
                    self.edge_colors.append((alpha, beta))
                self.regions.append(reordered_region)
                    

        self.vertices = np.array(self.vertices)

    
        vor.filtered_points = seeds
        vor.filtered_regions = regions
    
    
    def reconstruct_polygons(self):
        
        ## reconstruct with modified 
        
        return
    
    
    def plot_polygons(self, pic_size):

        image = Image.new("RGB", (pic_size[0], pic_size[1]))       
        draw = ImageDraw.Draw(image)
          
        # Add polygons 
        for i in range(len(self.regions)):
            reg = self.regions[i]
            poly = self.vertices[reg]
            region_id = self.region_colors[i][0]
            Rid = region_id//(255*255)
            Gid = (region_id - Rid*255*255)//255
            Bid = region_id - Rid*255*255 - Gid*255
            orientation = tuple([Rid, Gid, Bid])
            p = []

            poly = np.asarray(poly*pic_size[0], dtype=int) 
            for i in range(poly.shape[0]):
                p.append(tuple(poly[i]))
            
            draw.polygon(p, fill=orientation) 

        img = np.asarray(image)
        for i in range(pic_size[0]):
            for j in range(pic_size[1]):
                
                self.alpha_field[i,j] = img[i,j,0]*255*255+img[i,j,1]*255+img[i,j,2]
        
    
    
    def show_data_struct(self):
        
        
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        ax[0].scatter(self.vertices[:,0], self.vertices[:,1],s=5)
        ax[0].axis("equal")
        ax[0].set_xlim(0,1)
        ax[0].set_ylim(0,1)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        
        for x, y in self.edges:
            s = [self.vertices[x][0], self.vertices[y][0]]
            t = [self.vertices[x][1], self.vertices[y][1]]
            ax[1].plot(s, t, 'k')
        ax[1].axis("equal")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[2].imshow(self.color_choices[self.alpha_field], origin='lower', cmap='coolwarm')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        
        plt.savefig('./voronoi.png', dpi=400)
        




if __name__ == '__main__':

    size=100
    s = int(size/0.08)+1
    g1 = graph(size = (s, s), density = 2/size, noise=0.001/size, seed=1, BC='noflux')  
    g1.show_data_struct()     
'''
import pickle


with open('graph_data.pkl', 'wb') as outp:
    
    for i in range(5):
        g1 = graph(size = (125, 125), density = 0.2, noise=0.001)  
        pickle.dump(g1, outp, pickle.HIGHEST_PROTOCOL)
    del g1


with open('graph_data.pkl', 'rb') as inp:
      
   while True:
        try:
            g1 = pickle.load(inp)
            g1.show_data_struct()
        except EOFError:
            break
 
'''
    
    
    
    
