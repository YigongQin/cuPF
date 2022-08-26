import numpy as np
import scipy.io as sio
import h5py
import sys, os
from scipy import interpolate

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
temp_top = top + 50
constGR = False

Gmax = 0.5*0.1*float(sys.argv[1]);
Rmax = 2*1e4*float(sys.argv[2]);

if constGR:

    Gmin = Gmax
    Rmin = Rmax
else:
    Gmin = 2
    Rmin = 0.2*1e6    

ns = 21
G_list = np.linspace(Gmin, Gmax, num = ns)
R_list = np.linspace(Rmin, Rmax, num = ns)
print('R(x) uniform in space', R_list)

dh = top/(ns-1)
ave_speed = 0.5*(R_list[:-1]+R_list[1:])
t_sampled = np.zeros(ns)
t_sampled[1:] = dh/( ave_speed )
t_sampled = np.cumsum(t_sampled)
fR = interpolate.interp1d(t_sampled, R_list)
fG = interpolate.interp1d(t_sampled, G_list)

y0 = 2

x = np.linspace(0-BC,Lx+BC,nx)
y = np.linspace(0-BC,Ly+BC,ny)
#t = np.linspace(0,2*top/Rmax,nt)
t = np.linspace(0,t_sampled[-1],nt)
dt = t[1] - t[0]
print('input temperature time step', dt)
R_uni_t = fR(t)
G_uni_t = fG(t)
print('R(t) uniform in time', R_uni_t)


y_acc = np.zeros(nt)
y_acc[1:] = 0.5*(R_uni_t[:-1]+R_uni_t[1:])*dt
y_acc = np.cumsum(y_acc)
print('y(t)', y_acc)


add_nt = int(50/R_uni_t[-1]/dt)
print('additional sampled time step', add_nt)


T = np.zeros(nx*ny*(nt+add_nt))
alpha = np.zeros(nx*ny)
psi = np.zeros(nx*ny)
U = np.zeros(nx*ny)



for i in range(nx*ny*(nt+add_nt)):
    
    xi = i%nx
    yi = int( (i%(nx*ny))/nx )
    ti = int(i/(nx*ny))
    
#    T[i] = 933.3 + G*( y[yi] - 0.5*Rmax*(t[ti]**2/tmax) - y0)
    if ti<nt: 
        T[i] = G_uni_t[ti]*( y[yi] - y_acc[ti] - y0) 
    else: 
        T[i] = G_uni_t[nt-1]*( y[yi] - y_acc[nt-1] - y0 - (ti-nt+1)*R_uni_t[-1]*dt ) 
    if ti==0:
       psi[i] = y0 - y[yi]      
        
mac_folder = str(sys.argv[-1]) + 'line_AM/'    
if not os.path.exists(mac_folder):
    os.makedirs(mac_folder)

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
