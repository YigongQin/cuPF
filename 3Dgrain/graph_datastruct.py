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
import itertools

def angle_norm(angular):

    return - ( 2*(angular + pi/2)/(pi/2) - 1 )

eps = 1e-12
def in_bound(x, y):
    
    if x>=-eps and x<=1+eps and y>=-eps and y<=1+eps:
        return True
    else:
        return False
    

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
    return (x, y)


def periodic_move_p(p, pc):

    if p[0]<pc[0]-0.5-eps: p[0]+=1
    if p[0]>pc[0]+0.5+eps: p[0]-=1
    if p[1]<pc[1]-0.5-eps: p[1]+=1
    if p[1]>pc[1]+0.5+eps: p[1]-=1    



def periodic_dist_(p, pc):
    
    x,  y  = p
    xc, yc = pc
    
    if x<xc-0.5-eps: x+=1
    if x>xc+0.5+eps: x-=1
    if y<yc-0.5-eps: y+=1
    if y>yc+0.5+eps: y-=1         
           
    return (x-xc)**2 + (y-yc)**2



def relative_angle(p1, p2):
    
    p1 = periodic_move(p1, p2)
    
    return np.arctan2(p2[1]-p1[1], p2[0]-p1[0])


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
    def __init__(self, lxd: float = 20, seed: int = 1, noise: float = 0):
        mesh_size, grain_size = 0.08, 4
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
            self.random_voronoi()
            self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())
            self.alpha_pde = self.alpha_field
            self.update()
            self.num_regions = len(self.regions)
            self.num_vertices = len(self.vertices)
            
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
       
    def update(self):
        
        """
        Input: joint2vertex, vertices
        Output: edges, region_coors
        """
        
        
        self.vertex2joint = dict((v, k) for k, v in self.joint2vertex.items())
        self.vertex_neighbor.clear()                    
        self.edges.clear()
                    

        
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
        for region, verts in self.region_coors.items():
            if len(verts)<=1: continue
        #    assert len(verts)>1, ('one vertex is not a grain ', region, self.regions[region])
            
            moved_region = []

            vert_in_region = self.regions[region]
            tent_edge = set()
            
        
            def periodic_dist(nxt, cur):
                x,  y  = self.vertices[nxt]
                xc, yc = self.vertices[cur]
                
                if x<xc-0.5-eps: x+=1
                if x>xc+0.5+eps: x-=1
                if y<yc-0.5-eps: y+=1
                if y>yc+0.5+eps: y-=1                    
                return (x-xc)**2 + (y-yc)**2

            
       
            prev, cur = 0, 0  
            for i in range(len(vert_in_region)):   
                nxt_candidates = []
                for nxt in range(len(vert_in_region)):                        
                    if linked_edge_by_junction(self.vertex2joint[vert_in_region[cur]], \
                                               self.vertex2joint[vert_in_region[nxt]]) and nxt!=prev:
                        nxt_candidates.append(nxt)
                
           #     if len(nxt_candidates)==1:        
           #         prev, cur = cur, nxt_candidates[0]
                    
                if len(nxt_candidates)>1:
                    nxt_candidates = sorted(nxt_candidates, \
                                            key=lambda x: periodic_dist(vert_in_region[x], vert_in_region[cur]))
                    for nxt in nxt_candidates:
                        if (nxt, cur) in self.edges:
                            nxt_candidates[0] = nxt
                nxt = nxt_candidates[0] if len(nxt_candidates)>0 else prev
                prev, cur = cur, nxt            
                
                tent_edge.add((vert_in_region[prev], vert_in_region[cur]))        
                verts[cur] = periodic_move(verts[cur], verts[prev])                                

            
            if len(tent_edge)!=len(vert_in_region):
                print('found anam', vert_in_region, tent_edge)
                              
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
          #  print(sort_index, vert_in_region)
            counter_edge = set()
            for i in range(len(sorted_vert)):
                link = (sorted_vert[i-1], sorted_vert[i]) if i >0 else (sorted_vert[len(sorted_vert)-1], sorted_vert[i])
                
              #  if link in self.edges:
              #      print('already found one')

                
                counter_edge.add(link)
                
            if True: #len(counter_edge.intersection(tent_edge))<len(tent_edge)//2:
                for pair in tent_edge:
                  #  tent_edge.remove(pair)
                  #  tent_edge.add((pair[1], pair[0]))
                    self.edges.add((pair[1], pair[0]))
                    self.edges.add(pair)
            cnt += len(vert_in_region) 
            self.region_edge[region] = tent_edge
           # self.edges.update(tent_edge)
              #  self.edges.add((link[1],link[0]))
        print('num vertices of grains', cnt)
        print('num edges, junctions', len(self.edges), len(self.joint2vertex))        
        # form edge             

        for src, dst in self.edges:
            self.vertex_neighbor[src].add(dst)

        
        self.plot_polygons()
        self.compute_error_layer()


    def GNN_update(self, X: np.ndarray):
        
        for joint, coors in self.vertices.items():
            self.vertices[joint] = X[joint]
            
        self.update()

class graph_trajectory(graph):
    def __init__(self, lxd: float = 20, seed: int = 1, frames: int = 1, physical_params = {}):   
        super().__init__(lxd = lxd, seed = seed)
        
        self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())
        self.frames = frames # note that frames include the initial condition
        self.joint_traj = []
        self.edge_events = []
        self.grain_events = []
        
        self.show = False
        self.states = []
        self.physical_params = physical_params
        self.save_frame = [True]*self.frames

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
     
        self.num_vertex_features = 8  ## first 2 are x,y coordinates, next 5 are possible phase
        self.active_args = np.asarray(f['node_region'])
        self.active_args = self.active_args.\
            reshape((self.num_vertex_features, 5*len(self.vertices), data_frames ), order='F')
        self.active_coors = self.active_args[:2,:,:]
        self.active_max = self.active_args[2,:,:]
        self.active_args = self.active_args[3:,:,:]
        

        prev_joint = {k:[0,0,100] for k, v in self.joint2vertex.items()}
        all_grain = set(np.arange(self.num_regions)+1)
        
        for frame in range(self.frames):
           
            
            print('load frame %d'%frame)
            
            cur_joint = defaultdict(list)
            quadraples = defaultdict(list)
            for vertex in range(self.active_args.shape[1]): 
                max_neighbor = self.active_max[vertex, frame]
                args = set(self.active_args[:,vertex,frame])
                xp, yp = self.x[self.active_coors[0,vertex,frame]], self.y[self.active_coors[1,vertex,frame]]
                if -1 in args: args.remove(-1)
                if not args: continue
                args = tuple(sorted(args))
                
                if len(args)==4: 
                    if args not in quadraples or max_neighbor<quadraples[args][2]:    
                        quadraples[args] = [xp, yp, max_neighbor]
                   # quadraples.append([list(args),[xp, yp, max_neighbor]])
                    continue
                if len(args)>4:
                    print(colored('find more than qudraples', 'red'))
                
                if args not in cur_joint or max_neighbor<cur_joint[args][2]:    
                    cur_joint[args] = [xp, yp, max_neighbor]

            
            """
            deal with quadruples 
            
            """

            
            ## delete undetermined junctions from quadruples, add right ones later
            del_joints = []
            for q, coors in quadraples.items():
                q_list = list(q)
              #  print(q_list)
                for comb in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]:
                    arg = tuple([q_list[i] for i in comb])

                    if arg not in prev_joint and arg in cur_joint:

                        del_joints.append([arg, cur_joint[arg]])
                        del cur_joint[arg]
                        
            print('quadruples', len(quadraples))
            
            def check_connectivity(cur_joint):
               # jj_link = 0
               
                missing = set()
                miss_case = defaultdict(int)
                total_missing = 0
                for k1 in cur_joint.keys():
                    num_link = 0
                    for k2 in cur_joint.keys(): 
                        if k1!=k2 and len( set(k1).intersection(set(k2)) ) == 2:
                            num_link += 1
                #            jj_link += 1
                          #  print(jj_link, k1, k2)
                    if num_link !=3:

                        missing.update(set(k1))
                        miss_case.update({k1:3-num_link})
                        total_missing += abs(3-num_link)
                     #   print('find missing junction link', k1, 3-num_link)
                return total_missing, missing, miss_case   

            def miss_quadruple(quadraples):

                total_missing, missing, miss_case  = check_connectivity(cur_joint)
                print('total missing edges, ', total_missing)
                for q, coor in quadraples.items():
                    if set(q).issubset(missing):
                        
                        print('using quadraples',q,' to find missing link')
                        
                        possible = list(itertools.combinations(list(q), 3))
                        for c in miss_case.keys():
                            if c in possible:
                                possible.remove(c)
    
                        miss_case_sum = 0

                        for i, j in miss_case.items():
                            if len(set(i).intersection(set(q)))>=2:
                                miss_case_sum += j
                             #   max_case = max(max_case, j)                 
                            
                        print('np. missing links', miss_case_sum)        
                        max_case = 1 if miss_case_sum<4 else 2
                        for ans in list(itertools.combinations(possible, max_case)):
                            print('try ans', ans)
                            for a in ans:
                                cur_joint[a] = coor
                            cur, _, _ = check_connectivity(cur_joint)
                            if cur == total_missing -miss_case_sum:
                                print('fixed!')
                                total_missing = cur
                                break
                            else:
                                for a in ans:
                                        del cur_joint[a]
                                    
            miss_quadruple(quadraples)

            self.joint_traj.append(cur_joint)
            prev_joint = cur_joint
            


            # check loaded information
            
            self.alpha_pde = self.alpha_pde_frames[:,:,frame].T
            cur_grain, counts = np.unique(self.alpha_pde, return_counts=True)
            self.area_counts = dict(zip(cur_grain, counts))
            self.area_counts = {i:self.area_counts[i] if i in self.area_counts else 0 for i in range(self.num_regions)}
            cur_grain = set(cur_grain)
            
            
            grain_set = set()
            for k in cur_joint.keys():
                grain_set.update(set(k))



            eliminated_grains = all_grain - cur_grain
            
            all_grain = cur_grain


            if len(cur_joint)<2*len(cur_grain):
                for arg, coor in del_joints:
                    cur_joint[arg] = coor
                miss_quadruple(quadraples)
                
            
            if len(cur_joint)<2*len(cur_grain):
                print(colored('junction find failed', 'red'))

            
            print('number of grains in pixels %d'%len(cur_grain))
        #    print('number of grains junction %d'%len(grain_set))
            print('number of junctions %d'%len(cur_joint))
          #  print('estimated number of junction-junction links %d'%jj_link) 
            # when it approaches the end, 3*junction is not accurate
            for grain in eliminated_grains:
                for pair in self.region_edge[grain]:
                    self.edge_labels[(pair[0], pair[1])] = -100
                    self.edge_labels[(pair[1], pair[0])] = -100                                 
            
            self.vertex_matching(frame, cur_joint, eliminated_grains)
        
            self.update()
            self.form_states_tensor(frame)
            if self.error_layer>0.08:
                self.save_frame[frame] = False
            if len(self.edges)!=6*len(cur_grain):
                self.save_frame[frame] = False
                
            if self.show == True:
                self.show_data_struct()   
                
            print('====================================')  
            print('\n') 
              
                
    def vertex_matching(self, frame, cur_joint, eliminated_grains):
      
      
        print('\n')
        print('summary of event from frame %d to frame %d'%(frame-1, frame))
      #  cur_joint = self.joint_traj[frame]


        def add_edge_event(old_junction_i, old_junction_j):
            vert_old_i = self.joint2vertex[old_junction_i]
            vert_old_j = self.joint2vertex[old_junction_j]                
            switching_edges.add((vert_old_i, vert_old_j))
            switching_edges.add((vert_old_j, vert_old_i))
            self.edge_labels[(vert_old_i, vert_old_j)] = 1
            self.edge_labels[(vert_old_j, vert_old_i)] = 1  


        def quadruple_(junctions):
            quadraples = {}
            pairs =  set()
            for i in junctions:
                for j in junctions:
                    if len( set(i).difference(set(j)) )==1:
                        if (j, i) not in pairs:
                            pairs.add((i,j))
                            quadraples[tuple(sorted(set(i).union(set(j))))] = (i,j)            
            
            return quadraples
        
        
        def ns_last_vert(j1, i1, i2, q):

            j2 = set(q) - set(j1)
            j2.add(list(set(i1)-set(i2))[0])
            j2.add(list(set(i2)-set(i1))[0])

            return tuple(sorted(list(j2)))

        def perform_switching(old_junction_i, old_junction_j, new_junction_i, new_junction_j):
            
            
  
            self.joint2vertex[new_junction_i] = self.joint2vertex.pop(old_junction_i)
            self.joint2vertex[new_junction_j] = self.joint2vertex.pop(old_junction_j)
            
            print('neighor switching: ', old_junction_i, old_junction_j, ' --> ', new_junction_i, new_junction_j)
            
            if old_junction_i in old_joint: old_joint.remove(old_junction_i)
            if old_junction_j in old_joint: old_joint.remove(old_junction_j)
            if new_junction_i in new_joint: new_joint.remove(new_junction_i)
            if new_junction_j in new_joint: new_joint.remove(new_junction_j)    


        """
        
        E0: vertex moving
        
        """
        

        
        for k, v in cur_joint.items():
            cur_joint[k] = v[:2]

 
        
        old_vertices = self.vertices.copy()
        self.vertices.clear()

        print('Expect %d junctions removed'%(2*len(eliminated_grains)))
        
        
        
        print('\nE0:')
        
        def match():
            old_map = self.joint2vertex.copy()
            new_map = cur_joint.copy()
     
            for joint in self.joint2vertex.keys():
                if joint in cur_joint:        
                  
    
                 #   vert = self.joint2vertex[joint]
                 #   coors = cur_joint[joint]
                 #   self.vertices[vert] = periodic_move(coors, old_vertices[vert])
    
                    del old_map[joint]
                    del new_map[joint]
                    
            return old_map, new_map
                
        old_map, new_map = match()
        print('number of moving vertices', len(self.joint2vertex) - len(old_map))


        """
        
        E1: neighbor switching
        
        """          
        print('\nE1:')
        
        
        old = set(old_map.keys())
        new = set(new_map.keys())
        
        switching_edges = set() 
        

        
        if old!= new:
            
            
            old_joint = list(old-new)
            new_joint = list(new-old)
           # print('dispearing joints ', len(old_joint), ';  ' , old_joint)
          #  print('emerging joints', len(new_joint ), ';  ' , new_joint)
            
          #  assert len(old_joint) == len(new_joint), "lenght of old %d, new %d"%(len(old_joint), len(new_joint))
            
            quadraples = quadruple_(old_joint)
            quadraples_new = quadruple_(new_joint)
            
            switching_event = set(quadraples.keys()).intersection(set(quadraples_new.keys()))
          #  print(switching_event) 
                                
            for e2 in switching_event:
                
                old_junction_i, old_junction_j = quadraples[e2]
                new_junction_i, new_junction_j = quadraples_new[e2]
      
        
                old_i_x, old_j_x = old_vertices[self.joint2vertex[old_junction_i]], \
                                   old_vertices[self.joint2vertex[old_junction_j]] 
                new_i_x, new_j_x = cur_joint[new_junction_i][:2], cur_joint[new_junction_j][:2]                   
      
                
                
               # print(relative_angle(old_i_x, old_j_x), relative_angle(new_i_x, new_j_x))
                if abs(relative_angle(old_i_x, old_j_x) - relative_angle(new_i_x, new_j_x))>pi/2:
                #    print(colored('switch junction for less rotation', 'green'), new_junction_i, new_junction_j)
                    new_junction_i, new_junction_j = new_junction_j, new_junction_i
                    
                add_edge_event(old_junction_i, old_junction_j)
                perform_switching(old_junction_i, old_junction_j, new_junction_i, new_junction_j)
            
            
            quadraples = quadruple_(old_joint)
            quadraples_new = quadruple_(new_joint)
 
            """
            if left_over!= -1:
                for q, joints in quadraples_new.items():
                    for i in old_joint:
                        if set(i).issubset(set(q)):            
                            add_vert = ns_last_vert(i, joints[0], joints[1], q)
                            
                            self.joint2vertex[add_vert] = left_over
                            if i in old_joint: old_joint.remove(i)
                            if joints[0] in new_joint: new_joint.remove(joints[0])
                            if joints[1] in new_joint: new_joint.remove(joints[1])                           
                            perform_switching(add_vert, i, joints[0], joints[1])

            
            
            case = 0 
            for q, joints in quadraples.items():
                for j in new_joint:
                    if set(j).issubset(set(q)) and not case:
                        add_vert = ns_last_vert(j, joints[0], joints[1], q)
                       # print(old_joint, joints[0], joints[1])
                        add_edge_event(joints[0], joints[1])
                        perform_switching(joints[0], joints[1], j, add_vert)
                        old_joint.append(add_vert)
                        case = 1
            
            quadraples = quadruple_(old_joint)
            quadraples_new = quadruple_(new_joint)
            
            switching_event = set(quadraples.keys()).intersection(set(quadraples_new.keys()))

            for e2 in switching_event:
                
                old_junction_i, old_junction_j = quadraples[e2]
                new_junction_i, new_junction_j = quadraples_new[e2]

                perform_switching(old_junction_i, old_junction_j, new_junction_i, new_junction_j)    
            
                        

                        
                        
            if len(old_joint)>0:
                print(colored('match not finisehd','red'))
            
            """


        
        """
        
        E2: grain elimination
        
        """
        
        old_map, new_map = match()
      #  print(old_map, new_map )
        
        if len(eliminated_grains)>0:
            print('\nE2 grain_elimination: ', eliminated_grains)
            
        grain_grain_neigh = {}
        # step 1 merge grains to be eliminated
        for elm_grain in eliminated_grains:
            junction = set()
            for k, v in self.joint2vertex.items():
                if elm_grain in set(k):
                    junction.update(set(k))
            junction.remove(elm_grain) 
            grain_grain_neigh[elm_grain] = junction
      #  print(grain_grain_neigh)
        
        gg_merged = {}
        visited = set()
        for k1, v1 in grain_grain_neigh.items():
            ks, vs = [k1], v1
            for k2, v2 in grain_grain_neigh.items():
                if k1 != k2 and k2 not in visited:
                    if k1 in v2:
                        ks.append(k2)
                        vs.update(v2)
                        
                        visited.add(k2)
            if k1 not in visited:
                gg_merged[tuple(ks)] = vs
            visited.add(k1)            
            
            
     #   print(gg_merged)
      #  left_over = -1
        for elm_grain, junction in gg_merged.items():
 
            old_vert = []
            todelete = set()
            toadd = []
        #    junction = set()
            for k, v in self.joint2vertex.items():
                
                if len(set(elm_grain).intersection(set(k)))>0:
             #       junction.update(set(k))
                    old_vert.append(v)
                    todelete.add(k)
            
            for k, v in new_map.items():
                if set(k).issubset(junction):# and k not in self.joint2vertex:
                    
                    toadd.append(k)                    
                    
            if len(old_vert) == len(toadd) + 2:
                """ remove vertices connect to elim grains"""       
                print(elm_grain,'th grain eliminated with no. of sides %d'%len(todelete), junction)
                for k in todelete:
                    del self.joint2vertex[k]                     
                """ add vertices """       
                for i in range(len(toadd)):
                    self.joint2vertex[toadd[i]] = old_vert[i]
                    print('the new joint', toadd[i], 'inherit the vert', old_vert[i])
             #   del new_map[toadd[i]]
         #   assert len(old_vert) == 2
        
        """
        old_map, new_map = match()            
        for elm_grain, junction in gg_merged.items():
 
            old_vert = []
            todelete = set()
            toadd = []
        #    junction = set()
            for k, v in self.joint2vertex.items():
                
                if len(set(elm_grain).intersection(set(k)))>0:
             #       junction.update(set(k))
                    old_vert.append(v)
                    todelete.add(k)
      
            '''remove vertices connect to elim grains'''  
            print(elm_grain,'th grain eliminated with no. of sides %d'%len(todelete), junction)
            for k in todelete:
                del self.joint2vertex[k]   
                
            
            for k, v in new_map.items():
                if set(k).issubset(junction):# and k not in self.joint2vertex:      
                    toadd.append(k)                    
  
            
              #  left_over = old_vert[-1]
            diff = len(old_vert) - len(toadd) - 2
            neigh = []
            '''find missing vertices'''
            for k, v in new_map.items():
                if len( set(k).intersection(junction) ) == 2:
                    neigh.append([periodic_dist_(self.region_center[elm_grain[0]], v), k])
                    
            neigh = sorted(neigh)
            for i in range(diff):
                toadd.append(neigh[-i-1][1])
            
          #  print(toadd)
            ''' add vertices '''       
            for i in range(len(toadd)):
                self.joint2vertex[toadd[i]] = old_vert[i]
                print('the new joint', toadd[i], 'inherit the vert', old_vert[i])
                del new_map[toadd[i]]
            
        """
            
        self.grain_events.append(eliminated_grains)
        self.edge_events.append(switching_edges)    
        

        for joint in self.joint2vertex.keys():
            if joint in cur_joint:
                vert = self.joint2vertex[joint]
                coors = cur_joint[joint]
                self.vertices[vert] = periodic_move(coors, old_vertices[vert])
      
            else:
                vert = self.joint2vertex[joint]
                self.vertices[vert] = old_vertices[vert]
                print(colored('unmatched joint detected: ', 'red'), joint, self.joint2vertex[joint])
        for joint in cur_joint.keys():
            if joint not in self.joint2vertex:
                print(colored('unused joint detected: ', 'green'), joint)
      
        
       
        print('number of E2 %d, number of E1 %d'%(len(eliminated_grains), len(switching_edges)//2))
        
      

            

            

    def show_events(self):
        

        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot([len(i) for i in self.grain_events])
        ax.plot([len(i)//2 for i in self.edge_events])
        ax.set_xlabel('snapshot')
        ax.set_ylabel('# events')
        ax.legend(['grain elimination', 'neighbor switching'])
        
    def form_states_tensor(self, frame):
        
        hg = GrainHeterograph()
        grain_state = np.zeros((self.num_regions, len(hg.features['grain'])))
        joint_state = np.zeros((self.num_vertices, len(hg.features['joint'])))
        grain_mask = np.zeros((self.num_regions, 1), dtype=int)
        joint_mask = np.zeros((self.num_vertices, 1), dtype=int)
        
        s = self.imagesize[0]
        

        
        for grain, coor in self.region_center.items():
            grain_state[grain-1, 0] = coor[0]
            grain_state[grain-1, 1] = coor[1]
            grain_mask[grain-1, 0] = 1

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
            joint_mask[joint, 0] = 1
        
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
        
        
        hg.mask = {'grain':grain_mask, 'joint':joint_mask}
        
     #   joint_grain_neighbor = -np.ones((self.num_vertices,3), dtype=int)
      #  joint_joint_neighbor = -np.ones((self.num_vertices,3), dtype=int)
        for k, v in self.vertex_neighbor.items():
            if len(v)<3: print(colored('junction with less than three junction neighbor', 'red'))
            if len(v)>3: print(colored('junction with more than three junction neighbor', 'red'))
      #      joint_joint_neighbor[k][:len(v)] = np.array(list(v))
        
        hg.vertex2joint = self.vertex2joint
      #  for k, v in self.joint2vertex.items():
      #      joint_grain_neighbor[v] = np.array(list(k))
        
      #  hg.neighbor_dicts.update({('joint','joint'):joint_joint_neighbor})
      #  hg.neighbor_dicts.update({('joint','grain'):joint_grain_neighbor})
        
        hg.physical_params = self.physical_params
        hg.physical_params.update({'seed':self.seed})

        if frame>0:
            hg.edge_rotation = np.array(list(self.edge_labels.values()))

        
        self.states.append(hg) # states at current time

        # reset the edge_labels, check event at next snapshot
        self.edge_labels = {(src, dst):0 for src, dst in jj_edge} 

        
                
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
        
        self.targets_scaling = {'grain':10, 'joint':10}    
            
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
                (nxt.feature_dicts['joint'][:,:2] - self.feature_dicts['joint'][:,:2])
            
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
                (self.feature_dicts['joint'][:,:2] - prev.feature_dicts['joint'][:,:2])             
        
        self.feature_dicts['grain'][:,4] *= self.targets_scaling['grain']
        self.feature_dicts['grain'] = np.hstack((self.feature_dicts['grain'], prev_grad_grain))

        self.feature_dicts['joint'] = np.hstack((self.feature_dicts['joint'], prev_grad_joint)) 
                

        
        for nodes, features in self.features.items():
            self.features[nodes] = self.features[nodes] + self.features_grad[nodes]  
            assert len(self.features[nodes]) == self.feature_dicts[nodes].shape[1]
        
        
if __name__ == '__main__':


    parser = argparse.ArgumentParser("Generate heterograph data")
    parser.add_argument("--mode", type=str, default = 'check')
    parser.add_argument("--rawdat_dir", type=str, default = './')
    parser.add_argument("--train_dir", type=str, default = './sameGR/')
    parser.add_argument("--test_dir", type=str, default = './test/')
    parser.add_argument("--seed", type=int, default = 1)
    parser.add_argument("--level", type=int, default = 0)
    parser.add_argument("--frame", type=int, default = 13)
    args = parser.parse_args()
    args.train_dir = args.train_dir + 'level' + str(args.level) +'/'
    args.test_dir = args.test_dir + 'level' + str(args.level) +'/'
    """
    this script generates graph trajectory objects and training/testing data 
    the pde simulaion data is in rawdat_dir
    train_dir: processed data for training
    test_dir: processed data for testing
    seed: realization seed, each graph trajectory relates to one pde simulation
    level: 0: regression 
           1: regression + classification 
           2: regression + classification + mask 
    """
    
    
    
    if args.mode == 'train':
    
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
  
        for seed in [args.seed]:
            
            train_samples = []
            
            traj = graph_trajectory(seed = seed, frames = args.frame)
          #  traj.update()
          #  traj.show_data_struct()
      
            traj.load_trajectory(rawdat_dir = args.rawdat_dir)
            #traj.show_data_struct()
            
      
            for snapshot in range(traj.frames-1):
                """
                training data: snapshot -> snapshot + 1
                whether data is useful depends on both 
                <1> regression part:
                    grain exists at snapshot
                    triple junction exists at both (shift only)
                    
                <2> classification part:
                    edge exists at both (label 0)
                    edge switching (label 1)
                    unknown (mask out)
                    
                """
                if traj.save_frame[snapshot+1] == True:
                    
                    if ( args.level == 2 ) \
                    or ( args.level == 1 and len(traj.grain_events[snapshot+1])==0 ) \
                    or ( args.level == 0 and len(traj.grain_events[snapshot+1])==0 and \
                                             len(traj.edge_events[snapshot+1])==0 ):    
                        
                        hg = traj.states[snapshot]
                        hg.form_gradient(prev = None if snapshot ==0 else traj.states[snapshot-1], \
                                         nxt = traj.states[snapshot+1])
                        print('save frame %d -> %d, event level %d'%(snapshot, snapshot+1, args.level))
                        train_samples.append(hg)
                else:
                    print(colored('irregular data ignored, frame','red'), snapshot+1)
       
            with open(args.train_dir + 'case' + str(seed) + '.pkl', 'wb') as outp:
                dill.dump(train_samples, outp)

            with open(args.train_dir + 'traj' + str(seed) + '.pkl', 'wb') as outp:
                dill.dump(traj, outp)

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
        seed = 95
      #  g1 = graph(lxd = 20, seed=1) 
      #  g1.show_data_struct()
        traj = graph_trajectory(seed = seed, frames = 13)
        traj.load_trajectory(rawdat_dir = args.rawdat_dir)
    
    if args.mode == 'instance':
        
        for seed in range(20):
            print('\n')
            print('test seed', seed)
            try:
                g1 = graph(lxd = 20, seed=seed, noise = 0.01) 
            except:    
                print('seed %d failed with noise 0.01, try 0'%seed)
                g1 = graph(lxd = 20, seed=seed)

            g1.show_data_struct()                
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
    
