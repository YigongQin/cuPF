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
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from collections import defaultdict
import h5py
import glob, re

eps = 1e-12
def in_bound(x, y):
    
    if x>=-eps and x<=1+eps and y>=-eps and y<=1+eps:
        return True
    else:
        return False
    
def translate_min_dist(s, t):
    s0, s1 = s[0], t[0]
    t0 ,t1 = s[1], t[1]
    if s0 - s1 >0.5: s1 += 1
    if s1 - s0 >0.5: s0 += 1
    if t0 - t1 >0.5: t1 += 1
    if t1 - t0 >0.5: t0 += 1  
    return [s0 ,s1], [t0, t1]

def periodic_move(x, y, xc, yc):
    if x<xc-0.5: x+=1
    if x>xc+0.5: x-=1
    if y<yc-0.5: y+=1
    if y>yc+0.5: y-=1    
    
    return x, y

def hexagonal_lattice(dx=0.05, noise=0.0001, BC='periodic'):
    # Assemble a hexagonal lattice
    rows, cols = int(1/dx)+1, int(1/dx)
    print('cols and rows of grains: ', cols, rows)
    shiftx, shifty = 0.1*dx, 0.25*dx
    points = []
    in_points = []
    randNoise = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2)*noise, size=rows*cols*5)
    count = 0
    for row in range(rows*2):
        for col in range(cols):
            count+=1
            x = ( (col + (0.5 * (row % 2)))*np.sqrt(3) )*dx + shiftx
            y = row*0.5 *dx + shifty
         
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
                  points.append([x+1,y+1])
                  points.append([x-1,y-1])
                  points.append([x-1,y+1])
                  points.append([x+1,y-1])                     

    return points, in_points




        
class graph:
    def __init__(self, lxd, seed: int = 1, BC: str = 'periodic', randInit = True):
        mesh_size, grain_size = 0.08, 2
        self.lxd = lxd
        s = int(lxd/mesh_size)+1
        self.imagesize = (s, s)
        self.vertices = [] ## vertices coordinates
        self.vertex2region = defaultdict(set) ## (vertex index, x coordiante, y coordinate)  -> (region1, region2, region3)
        self.edges = []  ## index linkage
        self.edge_in_region = []  ## defined by region orientation 
        self.regions = [] ## index group
        self.region_coors = []
        self.region_label = []  ##color
        self.region_area = []
        self.region_center = []
       # self.region_coors = [] ## region corner coordinates
        self.density = grain_size/lxd
        self.noise = 0.001/lxd
        self.BC = BC
        self.alpha_field = np.zeros((self.imagesize[0], self.imagesize[1]), dtype=int)
        self.alpha_field_dummy = np.zeros((2*self.imagesize[0], 2*self.imagesize[1]), dtype=int)
        
        
        if randInit:
            np.random.seed(seed)
            self.random_voronoi()
            self.region_area = np.zeros(self.num_regions)
            self.plot_polygons(pic_size=self.imagesize)
            self.color_choices = np.zeros(2*self.num_regions+1)
            self.color_choices[1:] = -pi/2*np.random.random_sample(2*self.num_regions)
        
    def random_voronoi(self):

        mirrored_seeds, seeds = hexagonal_lattice(dx=self.density, noise = self.noise, BC = self.BC)
        vor = Voronoi(mirrored_seeds)     
    
       # regions = []
       # reordered_regions = []
       # vertices = []
        vert_map = {}
        vert_count = 0
        alpha = 0
       # edges = []
        
        for region in vor.regions:
            flag = True
            inboundpoints = 0
            upper_bound = 2 if self.BC == 'periodic' else 1
            for index in region:
                if index == -1:
                    flag = False
                    break
                else:
                    x = vor.vertices[index, 0]
                    y = vor.vertices[index, 1]
                    if x<=-eps or y<=-eps or x>=upper_bound+eps or y>=upper_bound+eps:
                        flag = False
                        break
                    if x<=1+eps and y<=1+eps:
                        inboundpoints +=1
                           
            
                        
            if region != [] and flag:
                polygon =  Polygon(vor.vertices[region]) 
                if inboundpoints ==0 and not polygon.contains(Point(1,1)): continue
                '''
                valid region propertities
                '''
                
            #    regions.append(region)
                reordered_region = []
                alpha += 1 
                self.region_label.append(alpha)
                
                for index in region:

                  #  point = tuple(vor.vertices[index])
                    x, y = round(vor.vertices[index][0]%1, 4), round(vor.vertices[index][1]%1, 4)
                    point = (x, y)
                    if point not in vert_map:
                        self.vertices.append(point)
                        vert_map[point] = vert_count
                        reordered_region.append(vert_count)
                        vert_count += 1
                    else:
                        reordered_region.append(vert_map[point])
                                          
                    
                for i in range(len(reordered_region)):
                    cur = self.vertices[reordered_region[i]]
                    nxt = self.vertices[reordered_region[i+1]] if i<len(reordered_region)-1 else self.vertices[reordered_region[0]]
                    cur, nxt = np.round(cur, 4), np.round(nxt, 4)
                    self.edges.append([vert_map[tuple(cur)], vert_map[tuple(nxt)]])
                    self.edge_in_region.append(alpha)
                    self.vertex2region[reordered_region[i]].add(alpha)  
                self.regions.append(reordered_region)
                self.region_coors.append(np.array(vor.vertices)[region])
                    

        self.vertices = np.array(self.vertices)
        self.num_regions = len(self.regions)
        self.num_vertices = self.vertices.shape[0]
            
    
      #  vor.filtered_points = seeds
      #  vor.filtered_regions = regions

    
        
    def plot_polygons(self, pic_size):

        image = Image.new("RGB", (2*pic_size[0], 2*pic_size[1]))       
        draw = ImageDraw.Draw(image)
          
        # Add polygons 
        for i in range(len(self.regions)):
            
            poly = self.region_coors[i]
            region_id = self.region_label[i]
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
        s = pic_size[0]
        for i in range(pic_size[0]):
            for j in range(pic_size[1]):
                ii, jj, = i ,j
                if img[i,j,2]==0:

                    if img[i+s,j,2]>0:
                        ii += s
                    elif img[i,j+s,2]>0:
                        jj += s
                    elif img[i+s,j+s,2]>0:
                        ii += s
                        jj += s
                    else:
                        #pass
                        raise ValueError(i,j)
                alpha = img[ii,jj,0]*255*255+img[ii,jj,1]*255+img[ii,jj,2]   
                self.alpha_field[i,j] = alpha 
                self.region_area[alpha-1] += 1
        for i in range(2*pic_size[0]):
            for j in range(2*pic_size[1]):
                alpha = img[i,j,0]*255*255+img[i,j,1]*255+img[i,j,2]   
                self.alpha_field_dummy[i,j] = alpha 
     
    def show_data_struct(self):
        
        
        
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        
        ax[0].scatter(self.vertices[:,0], self.vertices[:,1],s=5)
        ax[0].axis("equal")
        ax[0].set_title('Vertices'+str(self.num_vertices))
        #ax[0].set_xlim(0,1)
        #ax[0].set_ylim(0,1)
        #ax[0].set_xticks([])
        #ax[0].set_yticks([])
        
        for x, y in self.edges:
            s, t = translate_min_dist(self.vertices[x], self.vertices[y])
            ax[1].plot(s, t, 'k')
        ax[1].axis("equal")
        ax[1].set_title('Edges'+str(len(self.edges)))
        #ax[1].set_xticks([])
        #ax[1].set_yticks([])
        ax[2].imshow(self.color_choices[self.alpha_field+self.num_regions], origin='lower', cmap='coolwarm', vmin=-pi/2, vmax=0)
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_title('Grains'+str(self.num_regions))
        
        view_size = int(0.6*self.alpha_field_dummy.shape[0])
        ax[3].imshow(self.color_choices[self.alpha_field_dummy[:view_size,:view_size]+self.num_regions], origin='lower', cmap='coolwarm', vmin=-pi/2, vmax=0)
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        
        plt.savefig('./voronoi.png', dpi=400)
        


class graph_trajectory(graph):
    def __init__(self, seed):
        self.data_file = (glob.glob('*seed'+str(seed)+'*'))[0]
        f = h5py.File(self.data_file, 'r')
        self.x = np.asarray(f['x_coordinates'])
        self.y = np.asarray(f['y_coordinates'])
        self.z = np.asarray(f['z_coordinates']) 
        self.lxd = int(self.x[-2])
        self.x/=self.lxd; self.y/=self.lxd; self.z/=self.lxd
        number_list=re.findall(r"[-+]?\d*\.\d+|\d+", self.data_file)
        self.frames = int(number_list[2])+1
        super().__init__(lxd = self.lxd, seed = seed)
        
        self.num_vertex_features = 7
        self.active_args = np.asarray(f['node_region'])
        self.active_args = self.active_args.\
            reshape((self.num_vertex_features, 5*len(self.vertices), self.frames ), order='F')
        self.active_coors = self.active_args[:2,:,:]
        self.active_args = self.active_args[2:,:,:]
        self.region2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2region.items())
     
        self.vertex_xtraj = np.zeros((self.num_vertices, self.frames))
        self.vertex_ytraj = np.zeros((self.num_vertices, self.frames))
        self.outx = []
        self.outy = []
    def vertex_matching(self):
       
        active_vertices = set(np.arange(self.num_vertices))
     
        for frame in range(5):
            new_vertices = set()
            vertex_xmap, vertex_ymap = defaultdict(list), defaultdict(list)
            quadraples = {}
            for vertex in range(self.active_args.shape[1]): 
                args = set(self.active_args[:,vertex,frame])
                if -1 in args: args.remove(-1)
                if args: 
                    args = sorted(args)
                    if len(args)==4:
                        
                        if tuple(args) in quadraples:
                            combination = quadraples[tuple(args)]
                        else:
                        
                            print('find quadraple: ', args)
                           
                            touching_vert = []
                            for k in self.region2vertex.keys():
                                if set(k).issubset(args):
                                    touching_vert.append(k)
                            print('touching vertices',touching_vert, \
                                  self.vertex_xtraj[self.region2vertex[touching_vert[0]],frame-1],\
                                  self.vertex_xtraj[self.region2vertex[touching_vert[1]],frame-1],\
                                  self.vertex_ytraj[self.region2vertex[touching_vert[0]],frame-1],\
                                  self.vertex_ytraj[self.region2vertex[touching_vert[1]],frame-1]                                     
                                      )
                            vert1, vert2 = touching_vert
                            vert1, vert2 = set(vert1), set(vert2)
                            joint = list(vert1.intersection(vert2))
                            union = vert1.union(vert2)
                            newvert1, newvert2 = list(union), list(union)
                            newvert1.remove(joint[0]); newvert2.remove(joint[1])
                            combination = [ tuple(sorted(newvert1)),\
                                            tuple(sorted(newvert2)) ]
                            print('vertices neighbor switched', combination)
                            self.region2vertex[combination[0]] = self.region2vertex.pop(touching_vert[0])
                            self.region2vertex[combination[1]] = self.region2vertex.pop(touching_vert[1])
                            quadraples[tuple(args)] = combination
                        
                    else:
                        combination = [tuple(args)]
                        for k in quadraples.keys():
                            if set(args).issubset(set(k)):
                                combination = []
                    for subargs in combination:

                        try:
                            vert = self.region2vertex[subargs]
                        except:
                            raise KeyError('cannot find matched vertices for tuple' , subargs, 'at time: ', frame)
                            
                        new_vertices.add(vert)
                        xp, yp = self.x[self.active_coors[0,vertex,frame]], self.y[self.active_coors[1,vertex,frame]]
                        if frame>0:
                            xp, yp = periodic_move(xp, yp, \
                            self.vertex_xtraj[vert, frame-1], self.vertex_ytraj[vert, frame-1])
                        vertex_xmap[vert].append(xp)
                        vertex_ymap[vert].append(yp)
                      #  self.outx.append(self.x[self.active_coors[0,vertex,frame]])
                      #  self.outy.append(self.y[self.active_coors[1,vertex,frame]])
            print("time ", frame, "eliminated vertices ", active_vertices.difference(new_vertices))
            active_vertices = new_vertices
            print('current number of vertices %d'%len(active_vertices))

            for i in active_vertices:
                cluster_x, cluster_y = vertex_xmap[i], vertex_ymap[i]
                
               # print(cluster_x, i, self.vertex2region[i])
 
                self.vertex_xtraj[i, frame] = sum(cluster_x)/len(cluster_x) 
                self.vertex_ytraj[i, frame] = sum(cluster_y)/len(cluster_y) 

        self.vertices[:,0] = self.vertex_xtraj[:,frame]
        self.vertices[:,1] = self.vertex_ytraj[:,frame]
      #  print(vertex_visited)
      #  for i in list(vertex_visited):
        #    print(self.vertices[i])
if __name__ == '__main__':

    
    #g1 = graph(lxd = 10, seed=1)  
    #g1.show_data_struct()     
    
    traj = graph_trajectory(seed = 1)
    traj.show_data_struct()
    traj.vertex_matching()
    traj.show_data_struct()
    #print(traj.vertex_xtraj[:,0], traj.vertex_ytraj[:,0])
    
    # TODO:
    # 4) node matching and iteration for different time frames
    # 5) equi-spaced QoI sampling, change tip_y to tip_nz
    # 6) Image to graph qoi computation/ check sum to 1
    
    
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
    
    
    
    
