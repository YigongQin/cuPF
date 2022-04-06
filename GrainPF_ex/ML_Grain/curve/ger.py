import numpy as np
import scipy.io as sio
import h5py
import sys
from math import pi

nx = 1001
ny = 1001
nt = 11
Lx = 60
Ly = 60
BC = 1
top = 45


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

Radius = 32*2.5*2/pi


for i in range(nx*ny*nt):
    
    xi = i%nx
    yi = int( (i%(nx*ny))/nx )
    ti = int(i/(nx*ny))
    
    r = np.sqrt( (x[xi]-Lx)**2 + (y[yi]-Ly)**2 )

#    T[i] = 933.3 + G*( y[yi] - 0.5*Rmax*(t[ti]**2/tmax) - y0)
    T[i] = 933.3 + G*( Radius - r - Rmax*t[ti] ) 
    if ti==0:
       psi[i] = r - Radius    
        
if len(sys.argv) == 4: mac_folder = str(sys.argv[3]) + 'line_AM/'    
else: mac_folder = 'line_AM/'
np.savetxt(mac_folder+'x.txt', x, fmt='%1.4e',delimiter='\n') 
np.savetxt(mac_folder+'y.txt', y, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'t.txt', t, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'psi.txt', psi, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'U.txt', U, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'G.txt', np.asarray([G]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'Rmax.txt', np.asarray([Rmax*1e-6]), fmt='%1.4e',delimiter='\n')
hf = h5py.File(mac_folder+'Temp.h5', 'w')
hf.create_dataset('Temp', data=T)
hf.close()
