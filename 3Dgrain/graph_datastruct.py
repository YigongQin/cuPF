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
import h5py
import glob, re, math, os, argparse
from termcolor import colored
import dill

def angle_norm(angular):

    return - ( 2*(angular + pi/2)/(pi/2) - 1 )

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

def periodic_move(p, pc):
    x,  y  = p
    xc, yc = pc
    if x<xc-0.5: x+=1
    if x>xc+0.5: x-=1
    if y<yc-0.5: y+=1
    if y>yc+0.5: y-=1    
    
    return (x, y)

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
    def __init__(self, lxd: float = 10, seed: int = 1, BC: str = 'periodic', randInit: bool = True):
        mesh_size, grain_size = 0.08, 2
        self.lxd = lxd
        self.seed = seed
        s = int(lxd/mesh_size)+1
        self.imagesize = (s, s)
        self.vertices = defaultdict(list) ## vertices coordinates
        self.vertex2joint = defaultdict(set) ## (vertex index, x coordiante, y coordinate)  -> (region1, region2, region3)
        self.vertex_neighbor = defaultdict(set)
        self.edges = set()  ## index linkage
        self.edge_in_region = []  ## defined by region orientation 
        self.regions = defaultdict(list) ## index group
        self.region_coors = defaultdict(list)
        self.region_area = defaultdict(float)
        self.region_center = defaultdict(list)
       # self.region_coors = [] ## region corner coordinates
        self.density = grain_size/lxd
        self.noise = 0.001/lxd
        self.BC = BC
        self.alpha_field = np.zeros((self.imagesize[0], self.imagesize[1]), dtype=int)
        self.alpha_field_dummy = np.zeros((2*self.imagesize[0], 2*self.imagesize[1]), dtype=int)
        self.error_layer = 0
        
        if randInit:
            np.random.seed(seed)
            self.random_voronoi()
            self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())
            self.update()
            self.num_regions = len(self.regions)
            self.num_vertices = len(self.vertices)
            self.alpha_pde = self.alpha_field
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
                                          
                    
                for i in range(len(reordered_region)):

                    self.vertex2joint[reordered_region[i]].add(alpha)  

            
    
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
                        #pass
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
       
    def update(self):
        
        """
        Input: joint2vertex, vertices
        Output: edges, region_coors
        """
        
        # form edge
        self.vertex2joint = dict((v, k) for k, v in self.joint2vertex.items())
        self.vertex_neighbor.clear()
        for k1, v1 in self.joint2vertex.items():
            for k2, v2 in self.joint2vertex.items(): 
                if k1!=k2 and len( set(k1).intersection(set(k2)) ) >= 2:
                    self.vertex_neighbor[v1].add(v2) 
                    
        self.edges.clear()
        for k, v in self.vertex_neighbor.items():
            for i in v:
                self.edges.add((k, i))
        
        # form region
        self.regions.clear()
        self.region_coors.clear()
        self.region_center.clear()
        for k, v in self.joint2vertex.items():
            for region in set(k):
                self.regions[region].append(v)
                
                self.region_coors[region].append(self.vertices[v])
        
        for region, verts in self.region_coors.items():
        #    assert len(verts)>1, ('one vertex is not a grain ', region, self.regions[region])
            inbound = [verts[0][0]>-eps, verts[0][1]>-eps]
            moved_region = []
            for i in range(1, len(verts)):
                
                verts[i] = periodic_move(verts[i], verts[i-1])
                inbound = [i and (j>-eps) for i, j in zip(inbound, verts[i])]
            for vert in verts:
                vert = [i + 1*(not j) for i, j in zip(vert, inbound)]
                moved_region.append(vert)
            self.region_coors[region] = moved_region
            x, y = zip(*moved_region)
         
            self.region_center[region] = [np.mean(x), np.mean(y)]
            self.region_coors[region] = sorted(moved_region, \
                key = lambda x: counterclock(x, self.region_center[region]))        
                     
        
        self.plot_polygons()

    def GNN_update(self, X: np.ndarray):
        
        for joint, coors in self.vertices.items():
            self.vertices[joint] = X[joint]
            
        self.update()

class graph_trajectory(graph):
    def __init__(self, lxd: float = 10, seed: int = 1, frames: int = 1, physical_params = {}):   
        super().__init__(lxd = lxd, seed = seed)
        
        self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())
        self.frames = frames
        self.joint_traj = []
        self.events = np.zeros((2, self.frames), dtype=int)
        
        self.show = False
        self.states = []
        self.physical_params = physical_params

    def load_trajectory(self, rawdat_dir: str = './'):
       
        
        self.data_file = (glob.glob(rawdat_dir + '/*seed'+str(self.seed)+'_*'))[0]
        f = h5py.File(self.data_file, 'r')
        self.x = np.asarray(f['x_coordinates'])
        self.y = np.asarray(f['y_coordinates'])
        self.z = np.asarray(f['z_coordinates']) 

        assert int(self.lxd) == int(self.x[-2])
        
        self.x /= self.lxd; self.y /= self.lxd; self.z /= self.lxd
        fnx, fny = len(self.x), len(self.y)

        assert len(self.x) -2 == self.imagesize[0]
        assert len(self.y) -2 == self.imagesize[1]  
        
        number_list=re.findall(r"[-+]?\d*\.\d+|\d+", self.data_file)
        data_frames = int(number_list[2])+1
        
        self.physical_params = {'G':float(number_list[3]), 'R':float(number_list[4])}
        self.alpha_pde_frames = np.asarray(f['cross_sec'])
        self.alpha_pde_frames = self.alpha_pde_frames.reshape((fnx, fny, data_frames),order='F')[1:-1,1:-1,:]        
        
        self.extraV_frames = np.asarray(f['extra_area'])
        self.extraV_frames = self.extraV_frames.reshape((self.num_regions, data_frames), order='F')        
     
        self.num_vertex_features = 7  ## first 2 are x,y coordinates, next 5 are possible phase
        self.active_args = np.asarray(f['node_region'])
        self.active_args = self.active_args.\
            reshape((self.num_vertex_features, 5*len(self.vertices), data_frames ), order='F')
        self.active_coors = self.active_args[:2,:,:]
        self.active_args = self.active_args[2:,:,:]
        
        
 


        
        for frame in range(self.frames):
           
            cur_joint = defaultdict(list)
            quadraples = []
            for vertex in range(self.active_args.shape[1]): 
                args = set(self.active_args[:,vertex,frame])
                xp, yp = self.x[self.active_coors[0,vertex,frame]], self.y[self.active_coors[1,vertex,frame]]
                if -1 in args: args.remove(-1)
                if not args: continue
                if len(args)>3: 
                    quadraples.append([set(args),(xp, yp)])
                    continue
                
                args = tuple(sorted(args))
                cur_joint[args].append((xp,yp))

                      
            ## deal with quadraples here
            
            for q, coors in quadraples:
              #  print('qudraple: ', q)
                for k in self.joint2vertex.keys():
                    if set(k).issubset(q):
                   #     print('classified to triple', k)
                        cur_joint[k].append(coors)
                for k in cur_joint.keys():
                    if set(k).issubset(q):
                   #     print('classified to triple', k)
                        cur_joint[k].append(coors) 
                        
            self.joint_traj.append(cur_joint)
            
            
            # check loaded information
            
            self.alpha_pde = self.alpha_pde_frames[:,:,frame].T
            cur_grain = set(np.unique(self.alpha_pde))
            cur_covered_grain = set()
            for i in cur_joint.keys():
                cur_covered_grain.update(set(i))
            count = 0
            for k1, v1 in cur_joint.items():
                for k2, v2 in cur_joint.items(): 
                    if k1!=k2 and len( set(k1).intersection(set(k2)) ) >= 2:
                        count += 1
            
            print('====================================')
            print('load frame %d'%frame)
            print('number of grains %d'%len(cur_grain))
            print('number of region junctions covered %d'%len(cur_covered_grain))
            print('number of junctions %d'%len(cur_joint))
            print('number of links %d'%count)
            
            
        self.vertex_matching()
        
        
    def vertex_matching(self):
        
        # compare the cur_joint with the laset joint 
        
     
        all_grain = set(np.arange(self.num_regions)+1)
        for frame in range(self.frames):
            print('summary of frame %d'%frame)
            cur_joint = self.joint_traj[frame]
          #  self.alpha_field = self.alpha_pde_frames[:,:,frame].T
            self.alpha_pde = self.alpha_pde_frames[:,:,frame].T
            cur_grain, counts = np.unique(self.alpha_pde, return_counts=True)
            self.area_counts = dict(zip(cur_grain, counts))
            cur_grain = set(cur_grain)
            #cur_grain = set()
            #for i in cur_joint.keys():
            #    cur_grain.update(set(i))
            eliminated_grains = all_grain - cur_grain
            switching_event = set()
            all_grain = cur_grain
            if len(eliminated_grains)>0:
                print('E1 grain_elimination: ', eliminated_grains)
            
            
            new_vert = set()
            
            
            ###===================####
            
            # deal with eliminated grain
            
            ###==================###
            visited = set()
            for elm_grain in eliminated_grains:
                if elm_grain in visited:
                    continue
                old_vert = []
                todelete = set()
                junction = set()
                for k, v in self.joint2vertex.items():
                    if elm_grain in set(k):
                        junction.update(set(k))
                        old_vert.append(self.joint2vertex[k])
                        todelete.add(k)
                        

                '''        
                if len(junction) == 3:
                    
                    self.joint2vertex[tuple(sorted(junction))] = old_vert[-1]
                    print('the new joint', tuple(sorted(junction)), 'inherit the vert', old_vert[-1])
                    old_vert.pop()
                
                else:
                
                cur_junction = junction.copy()
                for neighbor in cur_junction:
                    if neighbor in eliminated_grains:
                        visited.add(neighbor)
                        for k, v in self.joint2vertex.items():
                            if neighbor in set(k):
                                junction.update(set(k))
                                if self.joint2vertex[k] not in old_vert:
                                    old_vert.append(self.joint2vertex[k])
                                todelete.add(k)
                 '''       
                junction.remove(elm_grain)        
                print('%dth grain eliminated with no. of sides %d'%(elm_grain, len(todelete)), junction)
                for k in todelete:
                    del self.joint2vertex[k]   
                    
                
                for k, v in cur_joint.items():
                    if set(k).issubset(junction) and k not in self.joint2vertex:
                        self.joint2vertex[k] = old_vert[-1]
                        print('the new joint', k, 'inherit the vert', old_vert[-1])
                        old_vert.pop()
                
             #   assert len(old_vert) == 2
                        
            ###===================####
            
            # deal with neighbor switching
            
            ###==================###            
            
            old = set(self.joint2vertex.keys())
            new = set(cur_joint.keys())
            if old!= new:
                pairs =  set()
                quadraples = {}
                old_joint = list(old-new)
                new_joint = list(new-old)
               # print('dispearing joints ', len(old_joint), ';  ' , old_joint)
              #  print('emerging joints', len(new_joint ), ';  ' , new_joint)
                
              #  assert len(old_joint) == len(new_joint), "lenght of old %d, new %d"%(len(old_joint), len(new_joint))
                
                for i in old_joint:
                    for j in old_joint:
                        if len( set(i).difference(set(j)) )==1:
                            if (j, i) not in pairs:
                                pairs.add((i,j))
                                quadraples[tuple(sorted(set(i).union(set(j))))] = (i,j)
             #   print(pairs, len(pairs))
             #   print(quadraples)

                pairs =  set()
                quadraples_new = {}
                for i in new_joint:
                    for j in new_joint:
                        if len( set(i).difference(set(j)) )==1:
                            if (j, i) not in pairs:
                                pairs.add((i,j))
                                quadraples_new[tuple(sorted(set(i).union(set(j))))] = (i,j)
              #  print(pairs, len(pairs))
              #  print(quadraples_new)     
                
                switching_event = set(quadraples.keys()).intersection(set(quadraples_new.keys()))
              #  print(switching_event)                      
                for e2 in switching_event:
                    old_junction_i, old_junction_j = quadraples[e2]
                    new_junction_i, new_junction_j = quadraples_new[e2]

                    
                    self.joint2vertex[new_junction_i] = self.joint2vertex.pop(old_junction_i)
                    self.joint2vertex[new_junction_j] = self.joint2vertex.pop(old_junction_j)
                    
                    print('E2 neighor switching: ', old_junction_i, old_junction_j, ' --> ', new_junction_i, new_junction_j)
                    
                
            old_vertices = self.vertices.copy()
            self.vertices.clear()
            for joint in self.joint2vertex.keys():
                if joint in cur_joint:
                    vert = self.joint2vertex[joint]
                    new_vert.add(vert)
                    coors = cur_joint[joint]
                    coors = [periodic_move(i, old_vertices[vert]) for i in coors]
                    cluster_x, cluster_y = zip(*coors)

                    self.vertices[vert] = (sum(cluster_x)/len(cluster_x), sum(cluster_y)/len(cluster_y))

                else:
                    vert = self.joint2vertex[joint]
                    self.vertices[vert] = old_vertices[vert]
                    print(colored('unmatched joint detected: ', 'red'), joint, self.joint2vertex[joint])
            for joint in cur_joint.keys():
                if joint not in self.joint2vertex:
                    print(colored('unused joint detected: ', 'green'), joint)

            
           
            print('number of E1 %d, number of E2 %d'%(len(eliminated_grains), len(switching_event)))
            print('====================================')
                    
            self.events[0,frame] = len(eliminated_grains)
            self.events[1,frame] = len(switching_event)
            self.update()
            self.form_states_tensor(frame)
            if self.show == True:
                self.show_data_struct()
            



    def show_events(self):
        

        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(self.events[0])
        ax.plot(self.events[1])
        ax.legend('grain elimination', 'neighbor switching')
        
    def form_states_tensor(self, frame):
        
        hg = GrainHeterograph()
        grain_state = np.zeros((self.num_regions, len(hg.features['grain'])))
        joint_state = np.zeros((self.num_vertices, len(hg.features['joint'])))
        
        
        s = self.imagesize[0]
        

        
        for grain, coor in self.region_center.items():
            grain_state[grain-1, 0] = coor[0]
            grain_state[grain-1, 1] = coor[1]

        grain_state[:, 2] = frame/self.frames

        grain_state[:, 3] = np.array(list(self.area_counts.values())) /s**2
        if frame>0:
            grain_state[:, 4] = self.extraV_frames[:, frame]/s**3
        
        
        grain_state[:, 5] = np.cos(self.theta_x[1:])
        grain_state[:, 6] = np.sin(self.theta_x[1:])
        grain_state[:, 7] = np.cos(self.theta_z[1:])
        grain_state[:, 8] = np.sin(self.theta_z[1:])
        
        
        for joint, coor in self.vertices.items():
            joint_state[joint, 0] = coor[0]
            joint_state[joint, 1] = coor[1]
        
        joint_state[:, 2] = frame/self.frames
        joint_state[:, 3] = 1 - np.log10(self.physical_params['G'])/2
        joint_state[:, 4] = self.physical_params['R']/2
        
        
        gj_edge = []
        for grains, joint in self.joint2vertex.items():
            for grain in grains:
                gj_edge.append([grain-1, joint])
        
        jg_edge = [[joint, grain] for grain, joint in gj_edge]
        jj_edge = [[src, dst] for src, dst in self.edges]
        
        hg.feature_dicts.update({'grain':grain_state})
        hg.feature_dicts.update({'joint':joint_state})
        hg.edge_index_dicts.update({hg.edge_type[0]:np.array(gj_edge).T})
        hg.edge_index_dicts.update({hg.edge_type[1]:np.array(jg_edge).T})
        hg.edge_index_dicts.update({hg.edge_type[2]:np.array(jj_edge).T})
        
        hg.physical_params = self.physical_params
        hg.physical_params.update({'seed':self.seed})
        
        self.states.append(hg)


        
                
class GrainHeterograph:
    def __init__(self):
        self.features = {'grain':['x', 'y', 'z', 'area', 'extraV', 'cosx', 'sinx', 'cosz', 'sinz'],
                         'joint':['x', 'y', 'z', 'G', 'R']}
        self.mask = {'grain':None, 'joint':None}
        
        self.features_grad = {'grain':['darea'], 'joint':['dx', 'dy']}
        
    
        self.targets = {'grain':['darea', 'extraV'], 'joint':['dx', 'dy'],}
                    #    'grain_event':'elimination', 'edge_event':'rotation'}    
        
        self.edge_type = [('grain', 'push', 'joint'), \
                          ('joint', 'pull', 'grain'), \
                          ('joint', 'connect', 'joint')]
            
        self.feature_dicts = {}
        self.target_dicts = {}
        self.edge_index_dicts = {}
        self.edge_weight_dicts = {}
        self.additional_feature_keys = {}
        
        self.physical_params = {}

    def form_gradient(self, prev, nxt):
        
        
        
        """
            
        Gradients for next prediction
            
        """        
        
        if nxt is not None:

        
            darea = nxt.feature_dicts['grain'][:,3:4] - self.feature_dicts['grain'][:,3:4]
            
            self.target_dicts['grain'] = np.hstack((darea, nxt.feature_dicts['grain'][:,4:5]))
                                         
            self.target_dicts['joint'] = nxt.feature_dicts['joint'][:,:2] - \
                                         self.feature_dicts['joint'][:,:2]
        
        #    self.target_dicts['grain_event'] = 1*( (nxt.mask['grain'] - self.mask['grain']) == 1 )
        
        #    self.target_dicts['edge_event'] = nxt.edge_rotation
        
                                        
        """
            
        Gradients of history
            
        """
        
        
                                     
        if prev is None:
            prev_grad_grain = 0*self.feature_dicts['grain'][:,:1]
            prev_grad_joint = 0*self.feature_dicts['joint'][:,:2]
                    
        else:
            prev_grad_grain = self.feature_dicts['grain'][:,3:4] - prev.feature_dicts['grain'][:,3:4] 
            prev_grad_joint = self.feature_dicts['joint'][:,:2] - prev.feature_dicts['joint'][:,:2]             
        
        self.feature_dicts['grain'] = np.hstack((self.feature_dicts['grain'], prev_grad_grain))

        self.feature_dicts['joint'] = np.hstack((self.feature_dicts['joint'], prev_grad_joint)) 
                

        
        for nodes, features in self.features.items():
            self.features[nodes] = self.features[nodes] + self.features_grad[nodes]  
            assert len(self.features[nodes]) == self.feature_dicts[nodes].shape[1]
        
        
if __name__ == '__main__':

    
    #g1 = graph(lxd = 10, seed=1)  
    #g1.show_data_struct()  
    parser = argparse.ArgumentParser("Generate heterograph data")
    parser.add_argument("--mode", type=str, default = 'check')
    parser.add_argument("--rawdat_dir", type=str, default = './')
    parser.add_argument("--train_dir", type=str, default = './data/')
    parser.add_argument("--test_dir", type=str, default = './test/')
    parser.add_argument("--seed", type=int, default = 1)
    args = parser.parse_args()
    
    
    if args.mode == 'train':
    
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
  
        for seed in [args.seed]:
            
            train_samples = []
            
            traj = graph_trajectory(seed = seed, frames = 4)
          #  traj.update()
          #  traj.show_data_struct()
      
            traj.load_trajectory(rawdat_dir = args.rawdat_dir)
           # traj.vertex_matching()
            #traj.show_data_struct()
            #print(traj.vertex_xtraj[:,0], traj.vertex_ytraj[:,0])
            
            
            # set gradient as output and add to features
            
            hg0 = traj.states[0]
            hg4 = traj.states[traj.frames - 1]
          #  print(hg0.feature_dicts['grain'][:,:1])
            hg0.form_gradient(prev = None, nxt = hg4)
            
            train_samples.append(hg0)
        
       
            with open(args.train_dir + 'case' + str(seed) + '.pkl', 'wb') as outp:
                dill.dump(train_samples, outp)

    if args.mode == 'test':   
        if not os.path.exists(args.test_dir):
            os.makedirs(args.test_dir) 
        
    # creating testing dataset
        for seed in [1]:
            
            test_samples = []
            
            traj = graph_trajectory(seed = seed, frames = 1, physical_params={'G':5, 'R':1})
            traj.load_trajectory(rawdat_dir = args.rawdat_dir)
            hg0 = traj.states[0]
            hg0.form_gradient(prev = None, nxt = None)
            test_samples.append(hg0)
          #  hg0.graph = graph(seed = seed)
            with open(args.test_dir + 'case' + str(seed) + '.pkl', 'wb') as outp:
                dill.dump(test_samples, outp)
     
        
    if args.mode == 'check':
        seed = 1
        g1 = graph(lxd = 20, seed=1) 
        g1.show_data_struct()
       # traj = graph_trajectory(seed = seed, frames = 25)
       # traj.load_trajectory(rawdat_dir = args.rawdat_dir)
    
    
    # TODO:
    # 4) node matching and iteration for different time frames
    # 5) equi-spaced QoI sampling, change tip_y to tip_nz
    # 6) Image to graph qoi computation/ check sum to 1
    # 7) run 2000 simulations and create one-to-one prediction
    # 8) 1 layer architecture and train loop
    
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
    
    
'''

vi, vj = self.joint2vertex[old_junction_i], self.joint2vertex[old_junction_j]
neib_i = self.vertex_neighbor[vi] - set([vj])
neib_j = self.vertex_neighbor[vj] - set([vi])

neib_i0, neib_i1 = list(neib_i)[0], list(neib_i)[1]
neib_j0, neib_j1 = list(neib_j)[0], list(neib_j)[1]
if len(set(self.vertex2joint[neib_i0]).intersection(set(self.vertex2joint[neib_j0])))==0:
    neib_j0, neib_j1 = neib_j1, neib_j0 

if not linked_edge_by_junction(self.vertex2joint[neib_i0], new_junction_i):
    new_junction_i, new_junction_j = new_junction_j, new_junction_i

new_vi, new_vj = self.joint2vertex[old_junction_i], self.joint2vertex[old_junction_j]   

self.vertex_neighbor[new_vi].remove(neib_i1)

self.vertex_neighbor[neib_i1].remove(new_vi)

self.vertex_neighbor[new_vi].add(neib_j0)
self.vertex_neighbor[neib_j0].add(new_vi)

self.vertex_neighbor[neib_j0].remove(new_vj)
self.vertex_neighbor[new_vj].remove(neib_j0)

self.vertex_neighbor[new_vj].add(neib_i1)
self.vertex_neighbor[neib_i1].add(new_vj)


'''

'''     
def vertex_matching(self):
active_vertices = set(np.arange(self.num_vertices))
new_vertices = set()
new_vertices.add(vert)
print("time ", frame, "eliminated vertices ", active_vertices.difference(new_vertices))
active_vertices = new_vertices
print('current number of vertices %d'%len(active_vertices))

            argset = set(args)
                if args in self.joint2vertex:
                vert = self.joint2vertex[args]
            else:
            
                for k in self.joint2vertex.keys():
                    if len(set(k).difference(argset))==1:
                        old_vert = set(k)
                union = argset.union(old_vert)
                
                for k in self.joint2vertex.keys():
                    if set(k).issubset(union) and set(k)!=set(old_vert):
                        touching_vert = set(k)
                
                joint = argset.intersection(old_vert, touching_vert)
                new_touching_vert = union.copy()
                new_touching_vert -= joint
                print(argset, new_touching_vert,old_vert, touching_vert)
                self.joint2vertex[args] = self.joint2vertex[tuple(sorted(old_vert))]
                self.joint2vertex[tuple(sorted(new_touching_vert))] = self.joint2vertex[tuple(sorted(touching_vert))]                     
            
                
if frame>0:
    xp, yp = periodic_move(xp, yp, \
    self.vertex_xtraj[vert, frame-1], self.vertex_ytraj[vert, frame-1])

return

'''    