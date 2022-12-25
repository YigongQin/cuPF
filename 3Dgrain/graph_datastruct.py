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
from matplotlib import cm
from matplotlib.colors import ListedColormap
coolwarm = cm.get_cmap('coolwarm', 256)
newcolors = coolwarm(np.linspace(0, 1, 256*100))
ly = np.array([255/256, 255/256, 255/256, 1])
newcolors[0, :] = ly
newcmp = ListedColormap(newcolors)
from scipy.spatial import Voronoi
from math import pi
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from collections import defaultdict
import math
import argparse


def angle_norm(angular):

    return - ( 2*(angular + pi/2)/(pi/2) - 1 )

eps = 1e-12
def in_bound(x, y):
    
    if x>=-eps and x<=1+eps and y>=-eps and y<=1+eps:
        return True
    else:
        return False
    


def periodic_move_p(p, pc):

    if p[0]<pc[0]-0.5-eps: p[0]+=1
    if p[0]>pc[0]+0.5+eps: p[0]-=1
    if p[1]<pc[1]-0.5-eps: p[1]+=1
    if p[1]>pc[1]+0.5+eps: p[1]-=1    



def periodic_move(p, pc):
    x,  y  = p
    xc, yc = pc
    """
    if x<xc-0.5-eps: x+=1
    if x>xc+0.5+eps: x-=1
    if y<yc-0.5-eps: y+=1
    if y>yc+0.5+eps: y-=1    
    """
    rel_x = x - xc
    rel_y = y - yc
    x += -1*(rel_x>0.5) + 1*(rel_x<-0.5) 
    y += -1*(rel_y>0.5) + 1*(rel_y<-0.5) 
    
    
    assert -0.5<x - xc<0.5
    assert -0.5<y - yc<0.5
    return [x, y]


def periodic_dist_(p, pc):
    
    x,  y  = p
    xc, yc = pc
    
    if x<xc-0.5-eps: x+=1
    if x>xc+0.5+eps: x-=1
    if y<yc-0.5-eps: y+=1
    if y>yc+0.5+eps: y-=1         
           
    return math.sqrt((x-xc)**2 + (y-yc)**2)





def linked_edge_by_junction(j1, j2):
    
    
    if len(set(j1).intersection(set(j2)))==2:
        return True
    else:
        return False
    

def counterclock(point, center):
    # Vector between point and the origin: v = p - o
    vector = [point[0]-center[0], point[1]-center[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    angle = math.atan2(vector[1], vector[0])
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector

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
    def __init__(self, lxd: float = 20, seed: int = 1, noise: float = 0.01):
        mesh_size, grain_size = 0.08, 4
        self.lxd = lxd
        self.seed = seed
        s = int(lxd/mesh_size)+1
        self.imagesize = (s, s)
        self.vertices = defaultdict(list) ## vertices coordinates
        self.vertex2joint = defaultdict(set) ## (vertex index, x coordiante, y coordinate)  -> (region1, region2, region3)
        self.vertex_neighbor = defaultdict(set)
        self.edges = []  ## index linkage
        self.edge_len = []
        self.regions = defaultdict(list) ## index group
        self.region_coors = defaultdict(list)
        self.region_edge = defaultdict(set)
        self.region_area = defaultdict(float)
        self.region_center = defaultdict(list)
       # self.region_coors = [] ## region corner coordinates
        self.density = grain_size/lxd
        self.noise = noise/lxd
        self.BC = 'periodic'
        self.alpha_field = np.zeros((self.imagesize[0], self.imagesize[1]), dtype=int)
        self.alpha_field_dummy = np.zeros((2*self.imagesize[0], 2*self.imagesize[1]), dtype=int)
        self.error_layer = 0
        
        randInit = True
        
        if randInit:
            np.random.seed(seed)
            
            try:
            
                self.random_voronoi()
                self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())
                self.alpha_pde = self.alpha_field
                self.update(init=True)
            
            except:
                self.noise = 0
                self.edges = [] 
                self.vertex2joint = defaultdict(set)
                self.random_voronoi()
                self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())
                self.alpha_pde = self.alpha_field
                self.update(init=True)            
            
            self.num_regions = len(self.regions)
            self.num_vertices = len(self.vertices)
            self.num_edges = len(self.edges)
            
            cur_grain, counts = np.unique(self.alpha_pde, return_counts=True)
            self.area_counts = dict(zip(cur_grain, counts))

            
            # sample orientations
            ux = np.random.randn(self.num_regions)
            uy = np.random.randn(self.num_regions)
            uz = np.random.randn(self.num_regions)
            self.theta_x = np.zeros(1 + self.num_regions)
            self.theta_z = np.zeros(1 + self.num_regions)
            self.theta_x[1:] = np.arctan2(uy, ux)%(pi/2)
            self.theta_z[1:] = np.arctan2(np.sqrt(ux**2+uy**2), uz)%(pi/2)

    
    def compute_error_layer(self):
        self.error_layer = np.sum(self.alpha_pde!=self.alpha_field)/len(self.alpha_pde.flatten())
        print('pointwise error at current layer: ', self.error_layer)
    
    def random_voronoi(self):

        mirrored_seeds, seeds = hexagonal_lattice(dx=self.density, noise = self.noise, BC = self.BC)
        vor = Voronoi(mirrored_seeds)     
    
       # regions = []
       # reordered_regions = []
       # vertices = []
        vert_map = {}
        vert_count = 0
        edge_count = 0
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

                
                for index in region:

                  #  point = tuple(vor.vertices[index])
                    x, y = round(vor.vertices[index][0]%1, 4), round(vor.vertices[index][1]%1, 4)
                    point = (x, y)
                    if point not in vert_map:
                        self.vertices[vert_count] = point
                        vert_map[point] = vert_count
                        reordered_region.append(vert_count)
                        vert_count += 1
                    else:
                        reordered_region.append(vert_map[point])
                                          
                sorted_vert = reordered_region    
                for i in range(len(reordered_region)):

                    self.vertex2joint[reordered_region[i]].add(alpha)  
                    
                    """
                    cur = sorted_vert[i]
                    nxt = sorted_vert[i+1] if i<len(sorted_vert)-1 else sorted_vert[0]
                    self.edges.update({edge_count:[cur, nxt]})
                    edge_count += 1
                    """
    
      #  vor.filtered_points = seeds
      #  vor.filtered_regions = regions
        
    def plot_polygons(self):
        """
        Input: region_coors
        Output: alpha_field, just index

        """
        
        
        s = self.imagesize[0]
        image = Image.new("RGB", (2*s, 2*s))       
        draw = ImageDraw.Draw(image)
          
        # Add polygons 
        for region_id, poly in self.region_coors.items():
            Rid = region_id//(255*255)
            Gid = (region_id - Rid*255*255)//255
            Bid = region_id - Rid*255*255 - Gid*255
            orientation = tuple([Rid, Gid, Bid])
            p = []

            #poly = np.asarray(poly*pic_size[0], dtype=int) 
            for i in range(len(poly)):
                coor = np.asarray(np.array(poly[i])*s, dtype=int)
                p.append(tuple(coor))
          #  print(p)
            if len(p)>1:
                draw.polygon(p, fill=orientation) 

        img = np.asarray(image)
        
        for i in range(s):
            for j in range(s):
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
                       # pass
                        raise ValueError(i,j)
                alpha = img[ii,jj,0]*255*255+img[ii,jj,1]*255+img[ii,jj,2]   
                self.alpha_field[i,j] = alpha 
                self.region_area[alpha] += 1
        
        for k, v in self.region_area.items():
            self.region_area[k]/=s**2
        
        for i in range(2*s):
            for j in range(2*s):
                alpha = img[i,j,0]*255*255+img[i,j,1]*255+img[i,j,2]   
                self.alpha_field_dummy[i,j] = alpha 
                
        
     
    def show_data_struct(self):
        

        
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        
        vertices = list(self.vertices.values())
        x, y = zip(*vertices)
        ax[0,0].scatter(list(x), list(y), s=5)
        ax[0,0].axis("equal")
        ax[0,0].set_title('Vertices'+str(len(self.vertex_neighbor)))
        #ax[0].set_xlim(0,1)
        #ax[0].set_ylim(0,1)
        #ax[0].set_xticks([])
        #ax[0].set_yticks([])
        
        for coors in self.region_coors.values():
            for i in range(len(coors)):
                cur = coors[i]
                nxt = coors[i+1] if i<len(coors)-1 else coors[0]
                ax[0,1].plot([cur[0],nxt[0]], [cur[1],nxt[1]], 'k')
                
        x, y = zip(*self.region_center.values())     
    
        ax[0,1].scatter(list(x), list(y), c = 'k')
        ax[0,1].axis("equal")
        ax[0,1].set_title('Edges'+str(len(self.edges)))
        #ax[1].set_xticks([])
        #ax[1].set_yticks([])

        
        view_size = int(0.6*self.alpha_field_dummy.shape[0])
        field = self.alpha_field_dummy[:view_size,:view_size]
        field = self.theta_z[field]
        ax[0,2].imshow((field/pi*180)*(field>0), origin='lower', cmap=newcmp, vmin=0, vmax=90)
        ax[0,2].set_xticks([])
        ax[0,2].set_yticks([])
        ax[0,2].set_title('Grains'+str(len(self.regions))) 

        ax[1,0].imshow(self.theta_z[self.alpha_field]/pi*180, origin='lower', cmap=newcmp, vmin=0, vmax=90)
        ax[1,0].set_xticks([])
        ax[1,0].set_yticks([])
        ax[1,0].set_title('vertex model reconstructed') 
        ax[1,1].imshow(self.theta_z[self.alpha_pde]/pi*180, origin='lower', cmap=newcmp, vmin=0, vmax=90)
        ax[1,1].set_xticks([])
        ax[1,1].set_yticks([])
        ax[1,1].set_title('pde')         
        
        ax[1,2].imshow(1*(self.alpha_pde!=self.alpha_field),cmap='Reds',origin='lower')
        ax[1,2].set_xticks([])
        ax[1,2].set_yticks([])
        ax[1,2].set_title('error'+'%d'%(self.error_layer*100)+'%')                 
               
        plt.savefig('./voronoi.png', dpi=400)
       
    def update(self, init = False):
        
        """
        Input: joint2vertex, vertices, edges, 
        Output: region_coors
        """
        
        
      #  self.vertex2joint = dict((v, k) for k, v in self.joint2vertex.items())
        self.vertex_neighbor.clear()                    
   
        # form region
        self.regions.clear()
        self.region_coors.clear()
        self.region_center.clear()
        self.region_edge.clear()
        
        for k, v in self.joint2vertex.items():
            for region in set(k):
                self.regions[region].append(v)
                
                self.region_coors[region].append(self.vertices[v])
        cnt = 0
        edge_count = 0
        for region, verts in self.region_coors.items():
            if len(verts)<=1: continue
        #    assert len(verts)>1, ('one vertex is not a grain ', region, self.regions[region])
            
            moved_region = []

            vert_in_region = self.regions[region]
            grain_edge = set()

            prev, cur = 0, 0
            for i in range(len(vert_in_region)):
                
                for nxt in range(len(vert_in_region)): 
                    if nxt!=prev:
                        if init:
                            if linked_edge_by_junction(self.vertex2joint[vert_in_region[cur]], \
                                                    self.vertex2joint[vert_in_region[nxt]]):
                               break
                          
                        else:
                            if [vert_in_region[cur], vert_in_region[nxt]] in self.edges:
                                break
                            
                prev, cur = cur, nxt
                verts[cur] = periodic_move(verts[cur], verts[prev]) 

                
            inbound = [True, True]
            
            for vert in verts:
                inbound = [i and (j>-eps) for i, j in zip(inbound, vert)]  
            for vert in verts:
                vert = [i + 1*(not j) for i, j in zip(vert, inbound)]
                moved_region.append(vert)
        
            x, y = zip(*moved_region)
         
            self.region_center[region] = [np.mean(x), np.mean(y)]
            
            
            
            sort_index = sorted(range(len(moved_region)), \
                key = lambda x: counterclock(moved_region[x], self.region_center[region]))  
            
                
            self.region_coors[region] = [moved_region[i] for i in sort_index]
            
            sorted_vert = [vert_in_region[i] for i in sort_index]
            self.regions[region] = sorted_vert


            cnt += len(vert_in_region) 

            
            for i in range(len(sorted_vert)):
                cur = sorted_vert[i]
                nxt = sorted_vert[i+1] if i<len(sorted_vert)-1 else sorted_vert[0]
                
                grain_edge.add((cur, nxt))
                
                if init:
                    self.edges.append([cur, nxt])
                    edge_count += 1
                    
            self.region_edge[region] = grain_edge
           # self.edges.update(tent_edge)
              #  self.edges.add((link[1],link[0]))
        print('num vertices of grains', cnt)
        print('num edges, junctions', len([i for i in self.edges if i[0]>-1 ]), len(self.joint2vertex))        
        # form edge             

        for src, dst in self.edges:
            if src>-1:
                self.vertex_neighbor[src].add(dst)
                if src not in self.vertices:
                    print('in', self.vertex2joint[src])
                    print(src, 'not available')
                if dst not in self.vertices:
                    print(dst, 'not available',src) 
                    print('in', self.vertex2joint[dst])
                                       
                self.edge_len.append(periodic_dist_(self.vertices[src], self.vertices[dst]))   
            
            

        if init:  
            self.plot_polygons()
        self.compute_error_layer()


    def GNN_update(self, X: np.ndarray):
        
        for joint, coors in self.vertices.items():
            self.vertices[joint] = X[joint]
            
        self.update()


        
                
class GrainHeterograph:
    def __init__(self):
        self.features = {'grain':['x', 'y', 'z', 'area', 'extraV', 'cosx', 'sinx', 'cosz', 'sinz'],
                         'joint':['x', 'y', 'z', 'G', 'R']}
        self.mask = {}
        
        self.features_grad = {'grain':['darea'], 'joint':['dx', 'dy']}
        
    
        self.targets = {'grain':['darea', 'extraV'], 'joint':['dx', 'dy']}
        self.events = {'grain_event':'elimination', 'edge_event':'rotation'}    
        
        self.edge_type = [('grain', 'push', 'joint'), \
                          ('joint', 'pull', 'grain'), \
                          ('joint', 'connect', 'joint')]
        
        self.targets_scaling = {'grain':20, 'joint':5}    
            
        self.feature_dicts = {}
        self.target_dicts = {}
        self.edge_index_dicts = {}
        self.edge_weight_dicts = {}
        self.additional_features = {}
        self.neighbor_dicts = {}
        
        self.physical_params = {}

    def form_gradient(self, prev, nxt):
        
        
        
        """
            
        Gradients for next prediction
            
        """        
        
        if nxt is not None:

        
            darea = nxt.feature_dicts['grain'][:,3:4] - self.feature_dicts['grain'][:,3:4]
            
            self.target_dicts['grain'] = self.targets_scaling['grain']*\
                np.hstack((darea, nxt.feature_dicts['grain'][:,4:5]))
                                         
            self.target_dicts['joint'] = self.targets_scaling['joint']*\
               self.subtract(nxt.feature_dicts['joint'][:,:2], self.feature_dicts['joint'][:,:2], 'next')
               # (nxt.feature_dicts['joint'][:,:2] - self.feature_dicts['joint'][:,:2])
            self.target_dicts['edge_event'] = nxt.edge_rotation        
            
            self.additional_features['nxt'] = nxt.edge_index_dicts
            
            

            
            # check if the grain neighbor of the junction is the same
            for i in range(len(self.mask['joint'])):
                if self.mask['joint'][i,0] == 1:
                    if i in nxt.vertex2joint and set(self.vertex2joint[i]) == set(nxt.vertex2joint[i]):
                        pass
                    else:
                        self.mask['joint'][i,0] = 0
                      #  print('not matched', i, self.vertex2joint[i])
            print('maximum movement', np.max(np.absolute(self.mask['joint']*self.target_dicts['joint'])))
            print('maximum grain change', np.max(np.absolute(self.target_dicts['grain'])))
            assert np.all(self.mask['joint']*self.target_dicts['joint']>-1) \
               and np.all(self.mask['joint']*self.target_dicts['joint']<1)
            assert np.all(self.target_dicts['grain']>-1) and (np.all(self.target_dicts['grain']<1))

                                        
        """
            
        Gradients of history
            
        """
        
        
                                     
        if prev is None:
            prev_grad_grain = 0*self.feature_dicts['grain'][:,:1]
            prev_grad_joint = 0*self.feature_dicts['joint'][:,:2]
                    
        else:
            prev_grad_grain = self.targets_scaling['grain']*\
                (self.feature_dicts['grain'][:,3:4] - prev.feature_dicts['grain'][:,3:4]) 
            prev_grad_joint = self.targets_scaling['joint']*\
                self.subtract(self.feature_dicts['joint'][:,:2], prev.feature_dicts['joint'][:,:2], 'prev')
               # (self.feature_dicts['joint'][:,:2] - prev.feature_dicts['joint'][:,:2])             
        
        self.feature_dicts['grain'][:,4] *= self.targets_scaling['grain']
        self.feature_dicts['grain'] = np.hstack((self.feature_dicts['grain'], prev_grad_grain))

        self.feature_dicts['joint'] = np.hstack((self.feature_dicts['joint'], prev_grad_joint)) 
                

        
        for nodes, features in self.features.items():
            self.features[nodes] = self.features[nodes] + self.features_grad[nodes]  
            assert len(self.features[nodes]) == self.feature_dicts[nodes].shape[1]


    @staticmethod
    def subtract(b, a, loc):
        
        short_len = len(a)
        
        if loc == 'prev':
            return np.concatenate((b[:short_len,:]-a, 0*b[short_len:,:]), axis=0)
            
        if loc == 'next':
            return b[:short_len,:]-a


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Generate heterograph data")
    parser.add_argument("--mode", type=str, default = 'check')
    parser.add_argument("--seed", type=int, default = 1)
    parser.add_argument("--level", type=int, default = 0)

    args = parser.parse_args()        
        
    if args.mode == 'check':
        seed = 8
        g1 = graph(lxd = 20, seed=seed) 
        g1.show_data_struct()

    
    if args.mode == 'instance':
        
        for seed in range(300):
            print('\n')
            print('test seed', seed)

            g1 = graph(lxd = 20, seed=seed) 


          #  g1.show_data_struct() 
               



