import numpy as np
import scipy.io as sio
import h5py
import sys

nx = 10
ny = 41
nt = 11

BC = 1

x = np.linspace(0-BC,10+BC,nx)
y = np.linspace(0-BC,40+BC,ny)
t = np.linspace(0,6e-5,nt)


T = np.zeros(nx*ny*nt)
alpha = np.zeros(nx*ny)
psi = np.zeros(nx*ny)
U = np.zeros(nx*ny)

np.random.seed(int(sys.argv[1]))

G = 5;
Rmax = 1e6;
tmax = t[-1]
y0 = 2

angle1 = np.random.randint(1,11,size=1)
angle2 = np.random.randint(1,11,size=1)
split = np.random.randint(2,8,size=1)

angle1 = 1
angle2 = 7 
split = 4
for i in range(nx*ny*nt):
    
    xi = i%nx
    yi = int( (i%(nx*ny))/nx )
    ti = int(i/(nx*ny))
    
    T[i] = 920 + G*( y[yi] - 0.5*Rmax*(t[ti]**2/tmax) - y0)
    
    if ti==0:
       psi[i] = y0 - y[yi]      
       if xi>split:  
         alpha[i] = angle2
       else: alpha[i] = angle1 
        
mac_folder = 'line_AM/'    
np.savetxt(mac_folder+'x.txt', x, fmt='%1.4e',delimiter='\n') 
np.savetxt(mac_folder+'y.txt', y, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'t.txt', t, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'psi.txt', psi, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'U.txt', U, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'alpha.txt', alpha, fmt='%5d',delimiter='\n')
hf = h5py.File(mac_folder+'Temp.h5', 'w')
hf.create_dataset('Temp', data=T)
hf.close()
