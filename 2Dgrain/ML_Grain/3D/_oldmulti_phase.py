import numpy as np
import scipy.io as sio
import h5py
import sys
from graph_datastruct import graph

nx = 13
ny = 13
nz = 13
nt = 11
Lx = 10
Ly = 10
Lz = 40
BC = Lx/(nx-3) 
top = 30


G = float(sys.argv[1]);
Rmax = 1e6*float(sys.argv[2]);
z0 = 2

x = np.linspace(0-BC,Lx+BC,nx)
y = np.linspace(0-BC,Ly+BC,ny)
z = np.linspace(0-BC,Lz+BC,nz)
t = np.linspace(0,top/Rmax,nt)
tmax = t[-1]

T = np.zeros(nx*ny*nz*nt)
#alpha = np.zeros(nx*ny*nz)
psi = np.zeros(nx*ny*nz)
U = np.zeros(nx*ny*nz)

dx_dim = 0.08
gnx = int(Lx/dx_dim) + 1

g1 = graph(size = (gnx, gnx), density = 0.2, noise=0.001) 
print('input shape of alpha_field, ', g1.alpha_field.shape)
alpha = g1.alpha_field
NG = len(g1.regions)
theta = g1.color_choices
for i in range(nx*ny*nz*nt):
    
    xi = i%nx
    yi = int( (i%(nx*ny))/nx )
    zi = int( (i%(nx*ny*nz))/(nx*ny) )
    ti = int(i/(nx*ny*nz))
    
    #T[i] = 920 + G*( y[yi] - 0.5*Rmax*(t[ti]**2/tmax) - y0)
    T[i] = G*( z[zi] - Rmax*t[ti] - z0)    
    if i==nx*ny*nz*nt-1: print(T[i], G, z[zi], Rmax, t[ti], z0)
    if ti==0:
       psi[i] = z0 - z[zi]      
        
mac_folder = 'line_AM/'    
np.savetxt(mac_folder+'x.txt', x, fmt='%1.4e',delimiter='\n') 
np.savetxt(mac_folder+'y.txt', y, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'z.txt', z, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'t.txt', t, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'psi.txt', psi, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'U.txt', U, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'alpha.txt', alpha, fmt='%d',delimiter='\n')
np.savetxt(mac_folder+'theta.txt', theta, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'G.txt', np.asarray([G]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'Rmax.txt', np.asarray([Rmax*1e-6]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'NG.txt', np.asarray([NG]), fmt='%1.4e',delimiter='\n')
hf = h5py.File(mac_folder+'Temp.h5', 'w')
hf.create_dataset('Temp', data=T)
hf.close()
