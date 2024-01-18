import numpy as np
from scipy import io as sio
from math import pi
import sys
import os
from scipy.interpolate import interp1d as itp1d
from scipy.interpolate import interp2d as itp2d
from scipy.interpolate import griddata
from scipy.optimize import fsolve, brentq
import h5py




#macrodata = 'macrodata_Q200W_rb75.00um_ts0.50ms.mat'#
macrodata =  sys.argv[1]
dd = sio.loadmat(macrodata,squeeze_me=True)

# transient data
y1d = dd['z_1d'][:,-1]            # um
U1d = dd['Uc_1d'][:,-1]             # um
psi_1d = dd['op_psi_1d'][:,-1]
ztip = dd['ztip']
tmac = dd['t_macro']
Gt = dd['G_t'][2,:]
Rt = dd['R_t'][2,:]
print(ztip)

#plt.scatter(points[:,0],points[:,1])


mac_folder = 'WD_shallow/'

np.savetxt(mac_folder+'y.txt', y1d, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'t.txt', tmac, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'psi.txt', psi_1d, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'U.txt', U1d, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'Gt.txt', Gt, fmt='%1.4e',delimiter='\n')
np.savetxt(mac_folder+'Rt.txt', Rt, fmt='%1.4e',delimiter='\n')
#np.savetxt(mac_folder+'Temp.txt', T_arr.reshape((Nx*Ny*Nt),order='F'), fmt='%1.6e',delimiter='\n')

#dd.update({'psi0':psi0,'U0':U0,'points':points,'psi_value':psi_value,'U_value':U_value,'line_angle':line_angle,'line_xst':line_xst,'line_yst':line_yst,\
#'cent':cent,'R0':R0,'line_id':line_id,'add_dist':add_dist,'lens':lens})
#sio.savemat('new'+sys.argv[1],dd)

# check data status of 































