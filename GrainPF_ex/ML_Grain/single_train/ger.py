import numpy as np
import scipy.io as sio
import h5py
import sys

nx = 11
ny = 11
nt = 11
w0=float(sys.argv[3]);
Ng=8
asp = 4
Lx = w0*Ng
Ly = Lx*asp
BC = 1
top = 60


G = 0.5*0.1*float(sys.argv[1]);
Rmax = 2*1e4*float(sys.argv[2]);
y0 = 2

x = np.linspace(0-BC,Lx+BC,nx)
y = np.linspace(0-BC,Ly+BC,ny)
#t = np.linspace(0,2*top/Rmax,nt)
t = np.linspace(0,top/Rmax,nt)
tmax = t[-1]

T = np.zeros(nx*ny*nt)
alpha = np.zeros(nx*ny)
psi = np.zeros(nx*ny)
U = np.zeros(nx*ny)


for i in range(nx*ny*nt):
    
    xi = i%nx
    yi = int( (i%(nx*ny))/nx )
    ti = int(i/(nx*ny))
    
#    T[i] = 933.3 + G*( y[yi] - 0.5*Rmax*(t[ti]**2/tmax) - y0)
    T[i] = 933.3 + G*( y[yi] - Rmax*(t[ti]) - y0) 
    if ti==0:
       psi[i] = y0 - y[yi]      
        
mac_folder = str(sys.argv[-1]) + 'line_AM/'    
np.savetxt(mac_folder+'x.txt', x, fmt='%1.4e',delimiter='\n') 
np.savetxt(mac_folder+'y.txt', y, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'t.txt', t, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'psi.txt', psi, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'U.txt', U, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'G.txt', np.asarray([G]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'Rmax.txt', np.asarray([Rmax*1e-6]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'w0.txt', np.asarray([w0]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'Ng.txt', np.asarray([Ng]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'asp.txt', np.asarray([asp]), fmt='%1.4e',delimiter='\n')
hf = h5py.File(mac_folder+'Temp.h5', 'w')
hf.create_dataset('Temp', data=T)
hf.close()
