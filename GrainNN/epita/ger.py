import numpy as np
import scipy.io as sio
import h5py
import sys

nx = 11
ny = 101
nt = 101
w0=float(sys.argv[3]);
Ng=8
asp = 4
Lx = w0*Ng
Ly = Lx*asp
BC = 1
top = 60
temp_top = top + 20
constGR = True

Gmax = 0.5*0.1*float(sys.argv[1]);
Rmax = 2*1e4*float(sys.argv[2]);

if constGR:

    Gmin = Gmax
    Rmin = Rmax
else:
    Gmin = 2
    Rmin = 0.2*1e6    

G_list = np.linspace(Gmin, Gmax, num = nt)
R_list = np.linspace(Rmin, Rmax, num = nt)



y0 = 2

x = np.linspace(0-BC,Lx+BC,nx)
y = np.linspace(0-BC,Ly+BC,ny)
#t = np.linspace(0,2*top/Rmax,nt)
t = np.linspace(0,top/Rmax,nt)
tmax = t[-1]
dt = t[1] - t[0]

T = np.zeros(nx*ny*nt)
alpha = np.zeros(nx*ny)
psi = np.zeros(nx*ny)
U = np.zeros(nx*ny)

y_acc = np.zeros(nt)
y_acc[1:] = 0.5*(R_list[:-1]+R_list[1:])*dt
y_acc = np.cumsum(y_acc)
print(y_acc)

for i in range(nx*ny*nt):
    
    xi = i%nx
    yi = int( (i%(nx*ny))/nx )
    ti = int(i/(nx*ny))
    
#    T[i] = 933.3 + G*( y[yi] - 0.5*Rmax*(t[ti]**2/tmax) - y0)
    T[i] = G_list[ti]*( y[yi] - y_acc[ti] - y0) 
    if ti==0:
       psi[i] = y0 - y[yi]      
        
mac_folder = str(sys.argv[-1]) + 'line_AM/'    
np.savetxt(mac_folder+'x.txt', x, fmt='%1.4e',delimiter='\n') 
np.savetxt(mac_folder+'y.txt', y, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'t.txt', t, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'psi.txt', psi, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'U.txt', U, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'G.txt', G_list[-1:], fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'Rmax.txt', R_list[-1:]*1e-6, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'w0.txt', np.asarray([w0]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'Ng.txt', np.asarray([Ng]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'asp.txt', np.asarray([asp]), fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'top.txt', np.asarray([top]), fmt='%1.4e',delimiter='\n')
hf = h5py.File(mac_folder+'Temp.h5', 'w')
hf.create_dataset('Temp', data=T)
hf.close()
