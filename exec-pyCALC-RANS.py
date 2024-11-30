from scipy import sparse
import numpy as np
import cupy as cp
import sys
import time
from scipy.sparse import spdiags,linalg,eye

def setup_case():
   global  c_omega_1, c_omega_2, cmu, convergence_limit_eps, convergence_limit_k, convergence_limit_om, convergence_limit_pp, \
   convergence_limit_u, convergence_limit_v, convergence_limit_w, dist,fx, fy,imon,jmon,kappa,k_bc_east,k_bc_east_type, \
   k_bc_north,k_bc_north_type,k_bc_south, k_bc_south_type,k_bc_west,k_bc_west_type,kom,maxit, \
   ni,nj,nsweep_kom, nsweep_pp, nsweep_vel,  om_bc_east, om_bc_east_type, om_bc_north, om_bc_north_type, \
   om_bc_south, om_bc_south_type, om_bc_west, om_bc_west_type, p_bc_east, p_bc_east_type, \
   p_bc_north, p_bc_north_type, p_bc_south, p_bc_south_type, p_bc_west, p_bc_west_type, \
   prand_k,prand_omega,resnorm_p,resnorm_vel,restart,save,save_vtk_movie,scheme,scheme_turb,solver_pp,solver_vel, \
   solver_turb,sormax, u_bc_east, u_bc_east_type, u_bc_north, u_bc_north_type, u_bc_south, u_bc_south_type, u_bc_west, \
   u_bc_west_type, urfvis, urf_vel, urf_k, urf_p,urf_omega,v_bc_east, v_bc_east_type, v_bc_north, v_bc_north_type, \
   v_bc_south, v_bc_south_type,v_bc_west, v_bc_west_type,viscos, vol,vtk,vtk_save,vtk_file_name,x2d, xp2d, y2d, yp2d


   import numpy as np
   import sys


########### section 1 choice of differencing scheme ###########
   scheme='h'  #hybrid
   scheme_turb='h'  #hybrid upwind-central 

########### section 2 turbulence models ###########
   cmu=0.09
   kom = True
   c_omega_1= 5./9.
   c_omega_2=3./40.
   prand_omega=2.0
   prand_k=2.0

########### section 3 restart/save ###########
   restart = False
   save = True

########### section 4 fluid properties ###########
   viscos=1/5200

########### section 5 relaxation factors ###########
   urfvis=0.5
   urf_vel=0.5
   urf_k=0.5
   urf_p=1.0
   urf_omega=0.5

########### section 6 number of iteration and convergence criterira ###########
   maxit=20000
   min_iter=1
   sormax=1e-7

   solver_vel='direct'
   solver_pp='direct'
   solver_turb='direct'

   nsweep_vel=50
   nsweep_pp=50
   nsweep_kom=1
   convergence_limit_u=1e-6
   convergence_limit_v=1e-6
   convergence_limit_k=1e-6
   convergence_limit_om=1e-6
   convergence_limit_pp=5e-4

########### section 7 all variables are printed during the iteration at node ###########
   imon=0
   jmon=10

########### section 8 save data for post-processing ###########
   vtk=False
   save_all_files=False
   vtk_file_name='bound'

########### section 9 residual scaling parameters ###########
   uin=20
   resnorm_p=uin*y2d[1,-1]
   resnorm_vel=uin**2*y2d[1,-1]


########### Section 10 boundary conditions ###########

# boundary conditions for u
   u_bc_west=cp.ones(nj)
   u_bc_east=cp.zeros(nj)
   u_bc_south=cp.zeros(ni)
   u_bc_north=cp.zeros(ni)

   u_bc_west_type='n' 
   u_bc_east_type='n' 
   u_bc_south_type='d'
   u_bc_north_type='d'

# boundary conditions for v
   v_bc_west=cp.zeros(nj)
   v_bc_east=cp.zeros(nj)
   v_bc_south=cp.zeros(ni)
   v_bc_north=cp.zeros(ni)

   v_bc_west_type='n' 
   v_bc_east_type='n' 
   v_bc_south_type='d'
   v_bc_north_type='d'

# boundary conditions for p
   p_bc_west=cp.zeros(nj)
   p_bc_east=cp.zeros(nj)
   p_bc_south=cp.zeros(ni)
   p_bc_north=cp.zeros(ni)

   p_bc_west_type='n'
   p_bc_east_type='n'
   p_bc_south_type='n'
   p_bc_north_type='n'

# boundary conditions for k
   k_bc_west=cp.zeros(nj)
   k_bc_east=cp.zeros(nj)
   k_bc_south=cp.zeros(ni)
   k_bc_north=cp.zeros(ni)

   k_bc_west_type='n'
   k_bc_east_type='n'
   k_bc_south_type='d'
   k_bc_north_type='d'

# boundary conditions for omega
   om_bc_west=cp.zeros(nj)
   om_bc_east=cp.zeros(nj)

   xwall_s=0.5*(x2d[0:-1,0]+x2d[1:,0])
   ywall_s=0.5*(y2d[0:-1,0]+y2d[1:,0])
   dist2_s=(yp2d[:,0]-ywall_s)**2+(xp2d[:,0]-xwall_s)**2
   om_bc_south=10*6*viscos/0.075/dist2_s

   xwall_n=0.5*(x2d[0:-1,-1]+x2d[1:,-1])
   ywall_n=0.5*(y2d[0:-1,-1]+y2d[1:,-1])
   dist2_n=(yp2d[:,-1]-ywall_n)**2+(xp2d[:,-1]-xwall_n)**2
   om_bc_north=10*6*viscos/0.075/dist2_n

   om_bc_west_type='n'
   om_bc_east_type='n'
   om_bc_south_type='d'
   om_bc_north_type='d'

   return 



def modify_init(u2d,v2d,k2d,om2d,vis2d):
   
# set inlet field in entre domain
#  u2d=cp.repeat(u_bc_west[None,:], repeats=ni, axis=0)
   k2d=cp.ones((ni,nj))
   om2d=cp.ones((ni,nj))

   vis2d=k2d/om2d+viscos

   return u2d,v2d,k2d,om2d,vis2d

def modify_inlet():

   global y_rans,y_rans,u_rans,v_rans,k_rans,om_rans,uv_rans,k_bc_west,eps_bc_west,om_bc_west

   return u_bc_west,v_bc_west,k_bc_west,om_bc_west,u2d_face_w,convw

def modify_conv(convw,convs):

# since we are solving for fully-developed channel flow, we know that the convection terms are zero
   convs=cp.zeros((ni,nj))
   convw=cp.zeros((ni,nj))

   return convw,convs

def modify_u(su2d,sp2d):

   global file1

# add a driving pressure gradient term
   su2d= su2d+vol

# we know that the convection and diffusion term in the x direction are zero
   aw2d=cp.zeros((ni,nj))
   ae2d=cp.zeros((ni,nj))

# we know that for this flow the wall shear stress mustt be equal to one (since the driving pressure
# gradient is equal to one). We print it every iteration to see if it is one. When it reaches one it is
# a good indicator that the flow has converged

   tauw_south=viscos*cp.sum(as_bound*u2d[:,0])/x2d[-1,0]
   tauw_north=viscos*cp.sum(an_bound*u2d[:,-1])/x2d[-1,0]

   print(f"{'tau wall, south: '} {tauw_south:.3f},{'  tau wall, north: '} {tauw_north:.3f}")


   return su2d,sp2d

def modify_v(su2d,sp2d):

   return su2d,sp2d

def modify_p(su2d,sp2d):

   return su2d,sp2d

def modify_k(su2d,sp2d):

# we know that the convection and diffusion term in the x direction are zero
   aw2d=cp.zeros((ni,nj))
   ae2d=cp.zeros((ni,nj))

   return su2d,sp2d

def modify_om(su2d,sp2d):

# we know that the convection and diffusion term in the x direction are zero
   aw2d=cp.zeros((ni,nj))
   ae2d=cp.zeros((ni,nj))

   return su2d,sp2d

def modify_outlet(convw):

# since we are solving for fully-developed channel flow, we know that the convection terms are zero
   convw=cp.zeros((ni+1,nj))

   return convw

def fix_omega():

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d

def modify_vis(vis2d):

   return vis2d


def fix_k():

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d
from scipy import sparse
import numpy as np
import cupy as cp
import sys
import time
import pyamg
from scipy.sparse import spdiags,linalg,eye
import socket
# Convert 
def init():
   print('hostname: ',socket.gethostname())

# distance to nearest wall
   ywall_s=0.5*(y2d[0:-1,0]+y2d[1:,0])
   dist_s=yp2d-ywall_s[:,None]
   ywall_n=0.5*(y2d[0:-1,-1]+y2d[1:,-1])
   dist_n=ywall_n[:,None] -yp2d
   cp.asarray(dist_s)
   cp.asarray(dist_n)
   dist=cp.minimum(dist_s,dist_n)

#  west face coordinate
   xw=0.5*(x2d[0:-1,0:-1]+x2d[0:-1,1:])
   yw=0.5*(y2d[0:-1,0:-1]+y2d[0:-1,1:])

   del1x=((xw-xp2d)**2+(yw-yp2d)**2)**0.5
   del2x=((xw-cp.roll(xp2d, 1, axis=0))**2+(yw-cp.roll(yp2d, 1, axis=0))**2)**0.5
   fx=del2x/(del1x+del2x)

#  south face coordinate
   xs=0.5*(x2d[0:-1,0:-1]+x2d[1:,0:-1])
   ys=0.5*(y2d[0:-1,0:-1]+y2d[1:,0:-1])

   del1y=((xs-xp2d)**2+(ys-yp2d)**2)**0.5
   del2y=((xs-cp.roll(xp2d,1,axis=1))**2+(ys-cp.roll(yp2d,1,axis=1))**2)**0.5
   fy=del2y/(del1y+del2y)

   areawy=cp.diff(x2d,axis=1)
   areawx=-cp.diff(y2d,axis=1)

   areasy=-cp.diff(x2d,axis=0)
   areasx=cp.diff(y2d,axis=0)

   areaw=(areawx**2+areawy**2)**0.5
   areas=(areasx**2+areasy**2)**0.5

# volume approaximated as the vector product of two triangles for cells
   ax=cp.diff(x2d,axis=1)
   ay=cp.diff(y2d,axis=1)
   bx=cp.diff(x2d,axis=0)
   by=cp.diff(y2d,axis=0)

   areaz_1=0.5*cp.abs(ax[0:-1,:]*by[:,0:-1]-ay[0:-1,:]*bx[:,0:-1])

   ax=cp.diff(x2d,axis=1)
   ay=cp.diff(y2d,axis=1)
   areaz_2=0.5*cp.abs(ax[1:,:]*by[:,0:-1]-ay[1:,:]*bx[:,0:-1])

   vol=areaz_1+areaz_2

# coeff at south wall (without viscosity)
   as_bound=areas[:,0]**2/(0.5*vol[:,0])

# coeff at north wall (without viscosity)
   an_bound=areas[:,-1]**2/(0.5*vol[:,-1])

# coeff at west wall (without viscosity)
   aw_bound=areaw[0,:]**2/(0.5*vol[0,:])

   ae_bound=areaw[-1,:]**2/(0.5*vol[-1,:])

   return areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy,aw_bound,ae_bound,as_bound,an_bound,dist

def print_indata():

   print('////////////////// Start of input data ////////////////// \n\n\n')

   print('\n\n########### section 1 choice of differencing scheme ###########')
   print(f"{'scheme: ':<29}   {scheme}")
   print(f"{'scheme_turb: ':<29}   {scheme_turb}")

   print('\n\n########### section 2 turbulence models ###########')

   print(f"{'cmu: ':<29} {cmu}")
   print(f"{'kom: ':<29} {kom}")
   if kom:
      print(f"{'c_omega_1: ':<29} {c_omega_1:.3f}")
      print(f"{'c_omega_2: ':<29} {c_omega_2}")
      print(f"{'prand_k: ':<29} {prand_k}")
      print(f"{'prand_omega: ':<29} {prand_omega}")

   print('\n\n########### section 3 restart/save ###########')
   print(f"{'restart: ':<29} {restart}")
   print(f"{'save: ':<29} {save}")

   print('\n\n########### section 4 fluid properties ###########')
   print(f"{'viscos: ':<29} {viscos:.2e}")

   print('\n\n########### section 5 relaxation factors ###########')
   print(f"{'urfvis: ':<29} {urfvis}")

   print('\n\n########### section 6 number of iteration and convergence criterira ###########')
   print(f"{'sormax: ':<29} {sormax}")
   print(f"{'maxit: ':<29} {maxit}")
   print(f"{'solver_vel: ':<29} {solver_vel}")
   print(f"{'solver_turb: ':<29} {solver_turb}")
   print(f"{'nsweep_vel: ':<29} {nsweep_vel}")
   print(f"{'nsweep_pp: ':<29} {nsweep_pp}")
   print(f"{'nsweep_kom: ':<29} {nsweep_kom}")
   print(f"{'convergence_limit_u: ':<29} {convergence_limit_u}")
   print(f"{'convergence_limit_v: ':<29} {convergence_limit_v}")
   print(f"{'convergence_limit_pp: ':<29} {convergence_limit_pp}")
   print(f"{'convergence_limit_k: ':<29} {convergence_limit_k}")
   print(f"{'convergence_limit_om: ':<29} {convergence_limit_om}")

   print('\n\n########### section 7 all variables are printed during the iteration at node ###########')
   print(f"{'imon: ':<29} {imon}")
   print(f"{'jmon: ':<29} {jmon}")


   print('\n\n########### section 8 time-averaging ###########')


   print('\n\n########### section 9 residual scaling parameters ###########')
   print(f"{'resnorm_p: ':<29} {resnorm_p:.1f}")
   print(f"{'resnorm_vel: ':<29} {resnorm_vel:.1f}")


   print('\n\n########### Section 10 grid and boundary conditions ###########')
   print(f"{'ni: ':<29} {ni}")
   print(f"{'nj: ':<29} {nj}")
   print('\n')
   print('\n')

   print('------boundary conditions for u')
   print(f"{' ':<5}{'u_bc_west_type: ':<29} {u_bc_west_type}")
   print(f"{' ':<5}{'u_bc_east_type: ':<29} {u_bc_east_type}")
   if u_bc_west_type == 'd':
      print(f"{' ':<5}{'u_bc_west[0]: ':<29} {u_bc_west[0]}")
   if u_bc_east_type == 'd':
      print(f"{' ':<5}{'u_bc_east[0]: ':<29} {u_bc_east[0]}")


   print(f"{' ':<5}{'u_bc_south_type: ':<29} {u_bc_south_type}")
   print(f"{' ':<5}{'u_bc_north_type: ':<29} {u_bc_north_type}")

   if u_bc_south_type == 'd':
      print(f"{' ':<5}{'u_bc_south[0]: ':<29} {u_bc_south[0]}")
   if u_bc_north_type == 'd':
      print(f"{' ':<5}{'u_bc_north[0]: ':<29} {u_bc_north[0]}")

   print('------boundary conditions for v')
   print(f"{' ':<5}{'v_bc_west_type: ':<29} {v_bc_west_type}")
   print(f"{' ':<5}{'v_bc_east_type: ':<29} {v_bc_east_type}")
   if v_bc_west_type == 'd':
      print(f"{' ':<5}{'v_bc_west[0]: ':<29} {v_bc_west[0]}")
   if v_bc_east_type == 'd':
      print(f"{' ':<5}{'v_bc_east[0]: ':<29} {v_bc_east[0]}")


   print(f"{' ':<5}{'v_bc_south_type: ':<29} {v_bc_south_type}")
   print(f"{' ':<5}{'v_bc_north_type: ':<29} {v_bc_north_type}")

   if v_bc_south_type == 'd':
      print(f"{' ':<5}{'v_bc_south[0]: ':<29} {v_bc_south[0]}")
   if v_bc_north_type == 'd':
      print(f"{' ':<5}{'v_bc_north[0]: ':<29} {v_bc_north[0]}")

   print('------boundary conditions for k')
   print(f"{' ':<5}{'k_bc_west_type: ':<29} {k_bc_west_type}")
   print(f"{' ':<5}{'k_bc_east_type: ':<29} {k_bc_east_type}")
   if k_bc_west_type == 'd':
      print(f"{' ':<5}{'k_bc_west[0]: ':<29} {k_bc_west[0]}")
   if k_bc_east_type == 'd':
      print(f"{' ':<5}{'k_bc_east[0]: ':<29} {k_bc_east[0]}")
   
   
   print(f"{' ':<5}{'k_bc_south_type: ':<29} {k_bc_south_type}")
   print(f"{' ':<5}{'k_bc_north_type: ':<29} {k_bc_north_type}")
   
   if k_bc_south_type == 'd':
      print(f"{' ':<5}{'k_bc_south[0]: ':<29} {k_bc_south[0]}")
   if k_bc_north_type == 'd':
      print(f"{' ':<5}{'k_bc_north[0]: ':<29} {k_bc_north[0]}")
   
   print('------boundary conditions for omega')
   print(f"{' ':<5}{'om_bc_west_type: ':<29} {om_bc_west_type}")
   print(f"{' ':<5}{'om_bc_east_type: ':<29} {om_bc_east_type}")
   if om_bc_west_type == 'd':
      print(f"{' ':<5}{'om_bc_west[0]: ':<29} {om_bc_west[0]:.1f}")
   if om_bc_east_type == 'd':
      print(f"{' ':<5}{'om_bc_east[0]: ':<29} {om_bc_east[0]:.1f}")
   
   print(f"{' ':<5}{'om_bc_south_type: ':<29} {om_bc_south_type}")
   print(f"{' ':<5}{'om_bc_north_type: ':<29} {om_bc_north_type}")
   
   if om_bc_south_type == 'd':
      print(f"{' ':<5}{'om_bc_south[0]: ':<29} {om_bc_south[0]:.1f}")
   if om_bc_north_type == 'd':
      print(f"{' ':<5}{'om_bc_north[0]: ':<29} {om_bc_north[0]:.1f}")

   print('\n\n\n ////////////////// End of input data //////////////////\n\n\n')

   return 

def compute_face_phi(phi2d,phi_bc_west,phi_bc_east,phi_bc_south,phi_bc_north,\
    phi_bc_west_type,phi_bc_east_type,phi_bc_south_type,phi_bc_north_type):
   import numpy as np

   phi2d_face_w=cp.empty((ni + 1, nj))
   phi2d_face_s=cp.empty((ni, nj + 1))
   phi2d_face_w[0:-1,:]=fx*phi2d+(1-fx)*cp.roll(phi2d,1,axis=0)
   phi2d_face_s[:,0:-1]=fy*phi2d+(1-fy)*cp.roll(phi2d,1,axis=1)

# west boundary 
   phi2d_face_w[0,:]=phi_bc_west
   if phi_bc_west_type == 'n': 
# neumann
      phi2d_face_w[0,:]=phi2d[0,:]

# east boundary 
   phi2d_face_w[-1,:]=phi_bc_east
   if phi_bc_east_type == 'n': 
# neumann
      phi2d_face_w[-1,:]=phi2d[-1,:]
      phi2d_face_w[-1,:]=phi2d_face_w[-2,:]

# south boundary 
   phi2d_face_s[:,0]=phi_bc_south
   if phi_bc_south_type == 'n': 
# neumann
      phi2d_face_s[:,0]=phi2d[:,0]

# north boundary 
   phi2d_face_s[:,-1]=phi_bc_north
   if phi_bc_north_type == 'n': 
# neumann
      phi2d_face_s[:,-1]=phi2d[:,-1]
   
   return phi2d_face_w,phi2d_face_s

def dphidx(phi_face_w,phi_face_s):

   phi_w=phi_face_w[0:-1,:]*areawx[0:-1,:]
   phi_e=-phi_face_w[1:,:]*areawx[1:,:]
   phi_s=phi_face_s[:,0:-1]*areasx[:,0:-1]
   phi_n=-phi_face_s[:,1:]*areasx[:,1:]
   return (phi_w+phi_e+phi_s+phi_n)/vol

def dphidy(phi_face_w,phi_face_s):

   phi_w=phi_face_w[0:-1,:]*areawy[0:-1,:]
   phi_e=-phi_face_w[1:,:]*areawy[1:,:]
   phi_s=phi_face_s[:,0:-1]*areasy[:,0:-1]
   phi_n=-phi_face_s[:,1:]*areasy[:,1:]
   return (phi_w+phi_e+phi_s+phi_n)/vol

def coeff(convw,convs,vis2d,prand,scheme_local):

   visw=np.zeros((ni+1,nj))
   viss=np.zeros((ni,nj+1))
   vis_turb=(vis2d-viscos)/prand

   visw[0:-1,:]=fx*vis_turb+(1-fx)*np.roll(vis_turb,1,axis=0)+viscos
   viss[:,0:-1]=fy*vis_turb+(1-fy)*np.roll(vis_turb,1,axis=1)+viscos

   volw=np.ones((ni+1,nj))*1e-10
   vols=np.ones((ni,nj+1))*1e-10
   volw[1:,:]=0.5*np.roll(vol,-1,axis=0)+0.5*vol
   diffw=visw[0:-1,:]*areaw[0:-1,:]**2/volw[0:-1,:]
   vols[:,1:]=0.5*np.roll(vol,-1,axis=1)+0.5*vol
   diffs=viss[:,0:-1]*areas[:,0:-1]**2/vols[:,0:-1]

   if scheme_local == 'h':
      if iter == 0:
         print('hybrid scheme, prand=',prand)

      aw2d=np.maximum(convw[0:-1,:],diffw+(1-fx)*convw[0:-1,:])
      aw2d=np.maximum(aw2d,0.)

      ae2d=np.maximum(-convw[1:,:],np.roll(diffw,-1,axis=0)-np.roll(fx,-1,axis=0)*convw[1:,:])
      ae2d=np.maximum(ae2d,0.)

      as2d=np.maximum(convs[:,0:-1],diffs+(1-fy)*convs[:,0:-1])
      as2d=np.maximum(as2d,0.)

      an2d=np.maximum(-convs[:,1:],np.roll(diffs,-1,axis=1)-np.roll(fy,-1,axis=1)*convs[:,1:])
      an2d=np.maximum(an2d,0.)

   if scheme_local == 'c':
      if iter == 0:
         print('CDS scheme, prand=',prand)
      aw2d=diffw+(1-fx)*convw[0:-1,:]
      ae2d=np.roll(diffw,-1,axis=0)-np.roll(fx,-1,axis=0)*convw[1:,:]

      as2d=diffs+(1-fy)*convs[:,0:-1]
      an2d=np.roll(diffs,-1,axis=1)-np.roll(fy,-1,axis=1)*convs[:,1:]

   aw2d[0,:]=0
   ae2d[-1,:]=0
   as2d[:,0]=0
   an2d[:,-1]=0

   return aw2d,ae2d,as2d,an2d,su2d,sp2d

def bc(su2d,sp2d,phi_bc_west,phi_bc_east,phi_bc_south,phi_bc_north\
     ,phi_bc_west_type,phi_bc_east_type,phi_bc_south_type,phi_bc_north_type):

   su2d=np.zeros((ni,nj))
   sp2d=np.zeros((ni,nj))

#south
   if phi_bc_south_type == 'd':
      sp2d[:,0]=sp2d[:,0]-viscos*as_bound
      su2d[:,0]=su2d[:,0]+viscos*as_bound*phi_bc_south

#north
   if phi_bc_north_type == 'd':
      sp2d[:,-1]=sp2d[:,-1]-viscos*an_bound
      su2d[:,-1]=su2d[:,-1]+viscos*an_bound*phi_bc_north

#west
   if phi_bc_west_type == 'd':
      sp2d[0,:]=sp2d[0,:]-viscos*aw_bound
      su2d[0,:]=su2d[0,:]+viscos*aw_bound*phi_bc_west
#east
   if phi_bc_east_type == 'd':
      sp2d[-1,:]=sp2d[-1,:]-viscos*ae_bound
      su2d[-1,:]=su2d[-1,:]+viscos*ae_bound*phi_bc_east

   return su2d,sp2d

def conv(u2d,v2d,p2d_face_w,p2d_face_s):
#compute convection
   u2d_face_w,u2d_face_s=compute_face_phi(u2d,u_bc_west,u_bc_east,u_bc_south,u_bc_north,\
    u_bc_west_type,u_bc_east_type,u_bc_south_type,u_bc_north_type)
   v2d_face_w,v2d_face_s=compute_face_phi(v2d,v_bc_west,v_bc_east,v_bc_south,v_bc_north,\
    v_bc_west_type,v_bc_east_type,v_bc_south_type,v_bc_north_type)

   apw=np.zeros((ni+1,nj))
   aps=np.zeros((ni,nj+1))

   convw=-u2d_face_w*areawx-v2d_face_w*areawy
   convs=-u2d_face_s*areasx-v2d_face_s*areasy

#\\\\\\\\\\\\\\\\\ west face

# create ghost cells at east & west boundaries with Neumann b.c.
   p2d_e=p2d
   p2d_w=p2d
# duplicate last row and put it at the end
   p2d_e=np.insert(p2d_e,-1,p2d_e[-1,:],axis=0)
# duplicate row 0 and put it before row 0 (west boundary)
   p2d_w=np.insert(p2d_w,0,p2d_w[0,:],axis=0)

   dp=np.roll(p2d_e,-1,axis=0)-3*p2d_e+3*p2d_w-np.roll(p2d_w,1,axis=0)

#  apw[1:,:]=fx*np.roll(ap2d_vel,-1,axis=0)+(1-fx)*ap2d_vel
   apw[0:-1,:]=fx*ap2d_vel+(1-fx)*np.roll(ap2d_vel,1,axis=0)
   apw[-1,:]=1e-20
 
   dvelw=dp*areaw/4/apw

# boundaries (no corrections)
   dvelw[0,:]=0
   dvelw[-1,:]=0

   convw=convw+areaw*dvelw

#\\\\\\\\\\\\\\\\\ south face
# create ghost cells at north & south boundaries with Neumann b.c.
   p2d_n=p2d
   p2d_s=p2d
# duplicate last column and put it at the end
   p2d_n=np.insert(p2d_n,-1,p2d_n[:,-1],axis=1)
# duplicate first column and put it before column 0 (south boundary)
   p2d_s=np.insert(p2d_s,0,p2d_s[:,0],axis=1)

   dp=np.roll(p2d_n,-1,axis=1)-3*p2d_n+3*p2d_s-np.roll(p2d_s,1,axis=1)

#  aps[:,1:]=fy*np.roll(ap2d_vel,-1,axis=1)+(1-fy)*ap2d_vel
   aps[:,0:-1]=fy*ap2d_vel+(1-fy)*np.roll(ap2d_vel,1,axis=1)
   aps[:,-1]=1e-20
 
   dvels=dp*areas/4/aps

# boundaries (no corrections)
   dvels[:,0]=0
   dvels[:,-1]=0

   convs=convs+areas*dvels

# boundaries 
# west
   if u_bc_west_type == 'd':
      convw[0,:]=-u_bc_west*areawx[0,:]-v_bc_west*areawy[0,:]
# east
   if u_bc_east_type == 'd':
      convw[-1,:]=-u_bc_east*areawx[-1,:]-v_bc_east*areawy[-1,:]
# south
   if v_bc_south_type == 'd':
      convs[:,0]=-u_bc_south*areasx[:,0]-v_bc_south*areasy[:,0]
# north
   if v_bc_north_type == 'd':
      convs[:,-1]=-u_bc_north*areasx[:,-1]-v_bc_north*areasy[:,-1]

   return convw,convs

def solve_2d(phi2d,aw2d,ae2d,as2d,an2d,su2d,ap2d,tol_conv,nmax,solver_local):
   if iter == 0:
      print('solve_2d called')
      print('nmax',nmax)

   aw=np.matrix.flatten(aw2d)
   ae=np.matrix.flatten(ae2d)
   as1=np.matrix.flatten(as2d)
   an=np.matrix.flatten(an2d)
   ap=np.matrix.flatten(ap2d)
  
   m=ni*nj

   A = sparse.diags([ap, -an[0:-1], -as1[1:], -ae, -aw[nj:]], [0, 1, -1, nj, -nj], format='csr')

   su=np.matrix.flatten(su2d)
   phi=np.matrix.flatten(phi2d)

   res_su=np.linalg.norm(su)
   resid_init=np.linalg.norm(A*phi - su)

   phi_org=phi

# bicg (BIConjugate Gradient)
# bicgstab (BIConjugate Gradient STABilized)
# cg (Conjugate Gradient) - symmetric positive definite matrices only
# cgs (Conjugate Gradient Squared)
# gmres (Generalized Minimal RESidual)
# minres (MINimum RESidual)
# qmr (Quasi
   if solver_local == 'direct':
      if iter == 0:
         print('solver in solve_2d: direct solver')
      info=0
      resid=np.linalg.norm(A*phi - su)
      phi = linalg.spsolve(A,su)
   if solver_local == 'pyamg':
      if iter == 0:
         print('solver in solve_2d: pyamg solver')
      App = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
      res_amg = []
      phi = App.solve(su, tol=tol_conv, x0=phi, residuals=res_amg)
      info=0
      print('Residual history in pyAMG', ["%0.4e" % i for i in res_amg])
   if solver_local == 'cgs':
      if iter == 0:
         print('solver in solve_2d: cgs')
      phi,info=linalg.cgs(A,su,x0=phi, tol=tol_conv, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'cg':
      if iter == 0:
         print('solver in solve_2d: cg')
      phi,info=linalg.cg(A,su,x0=phi, tol=tol_conv, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'gmres':
      if iter == 0:
         print('solver in solve_2d: gmres')
      phi,info=linalg.gmres(A,su,x0=phi, tol=tol_conv, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'qmr':
      if iter == 0:
         print('solver in solve_2d: qmr')
      phi,info=linalg.qmr(A,su,x0=phi, tol=tol_conv, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'lgmres':
      if iter == 0:
         print('solver in solve_2d: lgmres')
      phi,info=linalg.lgmres(A,su,x0=phi, tol=tol_conv, atol=tol_conv,  maxiter=nmax)  # good
   if info > 0:
      print('warning in module solve_2d: convergence in sparse matrix solver not reached')

# compute residual without normalizing with |b|=|su2d|
   if solver_local != 'direct':
      resid=np.linalg.norm(A*phi - su)

   delta_phi=np.max(np.abs(phi-phi_org))

   phi2d=np.reshape(phi,(ni,nj))
   phi2d_org=np.reshape(phi_org,(ni,nj))

   if solver_local != 'pyamg':
      print(f"{'residual history in solve_2d: initial residual: '} {resid_init:.2e}{'final residual: ':>30}{resid:.2e}\
      {'delta_phi: ':>25}{delta_phi:.2e}")

# we return the initial residual; otherwise the solution is always satisfied (but the non-linearity is not accounted for)
   return phi2d,resid_init

def calcu(su2d,sp2d,p2d_face_w,p2d_face_s):
   if iter == 0:
      print('calcu called')
# b.c., sources, coefficients

# presssure gradient
   dpdx=dphidx(p2d_face_w,p2d_face_s)
   su2d=su2d-dpdx*vol

# modify su & sp
   su2d,sp2d=modify_u(su2d,sp2d)

   ap2d=aw2d+ae2d+as2d+an2d-sp2d

# under-relaxation
   ap2d=ap2d/urf_vel
   su2d=su2d+(1-urf_vel)*ap2d*u2d

   return su2d,sp2d,ap2d

def calcv(su2d,sp2d,p2d_face_w,p2d_face_s):
   if iter == 0:
      print('calcv called')
# b.c., sources, coefficients 

# presssure gradient
   dpdy=dphidy(p2d_face_w,p2d_face_s)
   su2d=su2d-dpdy*vol

# modify su & sp
   su2d,sp2d=modify_v(su2d,sp2d)

   ap2d=aw2d+ae2d+as2d+an2d-sp2d

# under-relaxation
   ap2d=ap2d/urf_vel
   su2d=su2d+(1-urf_vel)*ap2d*v2d

# ap2d will be used in calcp; store it as ap2d_vel
   ap2d_vel=ap2d

   return su2d,sp2d,ap2d,ap2d_vel

def calck(su2d,sp2d,k2d,om2d,vis2d,u2d_face_w,u2d_face_s,v2d_face_w,v2d_face_s):
# b.c., sources, coefficients 
   if iter == 0:
      print('calck_kom called')

# production term
   dudx=dphidx(u2d_face_w,u2d_face_s)
   dvdx=dphidx(v2d_face_w,v2d_face_s)

   dudy=dphidy(u2d_face_w,u2d_face_s)
   dvdy=dphidy(v2d_face_w,v2d_face_s)

   gen= (2.*(dudx**2+dvdy**2)+(dudy+dvdx)**2)
   vist=np.maximum(vis2d-viscos,1e-10)
   su2d=su2d+vist*gen*vol

   sp2d=sp2d-cmu*om2d*vol

# modify su & sp
   su2d,sp2d=modify_k(su2d,sp2d)

   ap2d=aw2d+ae2d+as2d+an2d-sp2d

# under-relaxation
   ap2d=ap2d/urf_k
   su2d=su2d+(1-urf_k)*ap2d*k2d

   return su2d,sp2d,gen,ap2d

def calcom(su2d,sp2d,om2d,gen):
   if iter == 0:
      print('calcom called')


#--------production term
   su2d=su2d+c_omega_1*gen*vol

#--------dissipation term
   sp2d=sp2d-c_omega_2*om2d*vol

# modify su & sp
   su2d,sp2d=modify_om(su2d,sp2d)

   ap2d=aw2d+ae2d+as2d+an2d-sp2d

# under-relaxation
   ap2d=ap2d/urf_vel
   su2d=su2d+(1-urf_omega)*ap2d*om2d

   return su2d,sp2d,ap2d

def calcp(pp2d,ap2d_vel):

   if iter == 0:
      print('calcp called')
# b.c., sources, coefficients and under-relaxation for pp2d

   apw=np.zeros((ni+1,nj))
   aps=np.zeros((ni,nj+1))

   pp2d=0
#----------simplec: multiply ap by (1-urf)
   ap2d_vel=np.maximum(ap2d_vel*(1.-urf_vel),1.e-20)

#\\\\\\\\\\\\\\\\ west face 
#  visw[0:-1,:,:]=fx*vis_turb+(1-fx)*np.roll(vis_turb,1,axis=0)+viscos
#  viss[:,0:-1,:]=fy*vis_turb+(1-fy)*np.roll(vis_turb,1,axis=1)+viscos

#  apw[1:,:]=fx*np.roll(ap2d_vel,-1,axis=0)+(1-fx)*ap2d_vel
   apw[0:-1,:]=fx*ap2d_vel+(1-fx)*np.roll(ap2d_vel,1,axis=0)
   apw[0,:]=1e-20
   dw=areawx**2+areawy**2
   aw2d=dw[0:-1,:]/apw[0:-1,:]
   ae2d=np.roll(aw2d,-1,axis=0)

#\\\\\\\\\\\\\\\\ south face 
#  aps[:,1:]=fy*np.roll(ap2d_vel,-1,axis=1)+(1-fy)*ap2d_vel
   aps[:,0:-1]=fy*ap2d_vel+(1-fy)*np.roll(ap2d_vel,1,axis=1)
   aps[:,0]=1e-20
   ds=areasx**2+areasy**2
   as2d=ds[:,0:-1]/aps[:,0:-1]
   an2d=np.roll(as2d,-1,axis=1)

   as2d[:,0]=0
   an2d[:,-1]=0
   aw2d[0,:]=0
   ae2d[-1,:]=0

   ap2d=aw2d+ae2d+as2d+an2d

# continuity error
#  su2d=convw[0:-1,:]-np.roll(convw[0:-1,:],-1,axis=0)+convs[:,0:-1]-np.roll(convs[:,0:-1],-1,axis=1)
   su2d=convw[0:-1,:]-convw[1:,:]+convs[:,0:-1]-convs[:,1:]

# set pp2d=0 in [0,0] tp make it non-singular
   as2d[0,0]=0
   an2d[0,0]=0
   aw2d[0,0]=0
   ae2d[0,0]=0
   ap2d[0,0]=1
#  su2d[0,0]=0

   return aw2d,ae2d,as2d,an2d,su2d,ap2d


def correct_u_v_p(u2d,v2d,p2d):
   if iter == 0:
      print('correct_u_v_p called')

# correct convections
#\\\\\\\\\\\\\ west face
   convw[1:-1,:]=convw[1:-1,:]+aw2d[0:-1,:]*(pp2d[1:,:]-pp2d[0:-1,:])

#\\\\\\\\\\\\\ south face
   convs[:,1:-1]=convs[:,1:-1]+as2d[:,0:-1]*(pp2d[:,1:]-pp2d[:,0:-1])

# correct p
   p2d=p2d+urf_p*(pp2d-pp2d[0,0])

# compute pressure correecion at faces (N.B. p_bc_west,, ... are not used since we impose Neumann b.c., everywhere)
   pp2d_face_w,pp2d_face_s=compute_face_phi(pp2d,p_bc_west,p_bc_east,p_bc_south,p_bc_north,\
        'n','n','n','n')

   dppdx=dphidx(pp2d_face_w,pp2d_face_s)
   u2d=u2d-dppdx*vol/ap2d_vel

   dppdy=dphidy(pp2d_face_w,pp2d_face_s)
   v2d=v2d-dppdy*vol/ap2d_vel


   return convw,convs,p2d,u2d,v2d,su2d


def vist_kom(vis2d,k2d,om2d):
   if iter == 0:
      print('vist_kom called')

   visold= vis2d
   vis2d= k2d/om2d+viscos

# modify viscosity
   vis2d=modify_vis(vis2d)

#            under-relax viscosity
   vis2d= urfvis*vis2d+(1.-urfvis)*visold

   return vis2d

def save_vtk():
   scalar_names = ['pressure']
   scalar_variables = [p2d]
   scalar_names.append('turb_kin')
   scalar_names.append('omega')
   scalar_variables.append(k2d)
   scalar_variables.append(om2d)

   if save_vtk_movie:
      file_name = '%s.%d.vtk' % (vtk_file_name, itstep)
   else:
      file_name = '%s.vtk' % (vtk_file_name)

   nk=1
   dz=1
   f = open(file_name,'w')
   f.write('# vtk DataFile Version 3.0\npyCALC-LES Data\nASCII\nDATASET STRUCTURED_GRID\n')
   f.write('DIMENSIONS %d %d %d\nPOINTS %d double\n' % (nk+1,nj+1,ni+1,(ni+1)*(nj+1)*(nk+1)))
   for i in range(ni+1):
      for j in range(nj+1):
         for k in range(nk+1):
            f.write('%.5f %.5f %.5f\n' % (x2d[i,j],y2d[i,j],dz*k))
   f.write('\nCELL_DATA %d\n' % (ni*nj*nk))

   f.write('\nVECTORS velocity double\n')
   for i in range(ni):
      for j in range(nj):
         for k in range(nk):
            f.write('%.12e %.12e %.12e\n' % (u2d[i,j,k],v2d[i,j,k],w2d[i,j,k]))

   for v in range(len(scalar_names)):
      var_name = scalar_names[v]
      var = scalar_variables[v]
      f.write('\nSCALARS %s double 1\nLOOKUP_TABLE default\n' % (var_name))
      for i in range(ni):
         for j in range(nj):
            for k in range(nk):
               f.write('%.10e\n' % (var[i,j,k]))
   f.close()

   print('Flow state save into VTK format to file %s\n' % (file_name))

def read_restart_data():

   print('read_restart_data called')

   u2d=np.load('u2d_saved.npy')
   v2d=np.load('v2d_saved.npy')
   p2d=np.load('p2d_saved.npy')
   k2d=np.load('k2d_saved.npy')
   om2d=np.load('om2d_saved.npy')
   vis2d=np.load('vis2d_saved.npy')

   return u2d,v2d,p2d,k2d,om2d,vis2d

def save_data(u2d,v2d,p2d,k2d,om2d,vis2d):

   print('save_data called')
   np.save('u2d_saved', u2d)
   np.save('v2d_saved', v2d)
   np.save('p2d_saved', p2d)
   np.save('k2d_saved', k2d)
   np.save('om2d_saved', om2d)
   np.save('vis2d_saved', vis2d)

   return 

######################### the execution of the code starts here #############################

########### grid specification ###########
datax= cp.loadtxt("x2d.dat")
x=datax[0:-1]
ni=int(datax[-1])
datay= np.loadtxt("y2d.dat")
y=datay[0:-1]
nj=int(datay[-1])

x2d=cp.zeros((ni+1,nj+1))
y2d=cp.zeros((ni+1,nj+1))

x2d=cp.reshape(x,(ni+1,nj+1))
y2d=cp.reshape(y,(ni+1,nj+1))

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])

# initialize geometric arrays

vol=cp.zeros((ni,nj))
areas=cp.zeros((ni,nj+1))
areasx=cp.zeros((ni,nj+1))
areasy=cp.zeros((ni,nj+1))
areaw=cp.zeros((ni+1,nj))
areawx=cp.zeros((ni+1,nj))
areawy=cp.zeros((ni+1,nj))
areaz=cp.zeros((ni,nj))
as_bound=cp.zeros((ni))
an_bound=cp.zeros((ni))
aw_bound=cp.zeros((nj))
ae_bound=cp.zeros((nj))
az_bound=cp.zeros((ni,nj))
fx=cp.zeros((ni,nj))
fy=cp.zeros((ni,nj))

setup_case()

print_indata()

areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy,aw_bound,ae_bound,as_bound,an_bound,dist=init()


# initialization
u2d=cp.ones((ni,nj))*1e-20
v2d=cp.ones((ni,nj))*1e-20
p2d=cp.ones((ni,nj))*1e-20
pp2d=cp.ones((ni,nj))*1e-20
k2d=cp.ones((ni,nj))*1
om2d=cp.ones((ni,nj))*1
vis2d=cp.ones((ni,nj))*viscos

fmu2d=cp.ones((ni,nj))
gen=cp.ones((ni,nj))

convw=cp.ones((ni+1,nj))*1e-20
convs=cp.ones((ni,nj+1))*1e-20

aw2d=cp.ones((ni,nj))*1e-20
ae2d=cp.ones((ni,nj))*1e-20
as2d=cp.ones((ni,nj))*1e-20
an2d=cp.ones((ni,nj))*1e-20
al2d=cp.ones((ni,nj))*1e-20
ah2d=cp.ones((ni,nj))*1e-20
ap2d=cp.ones((ni,nj))*1e-20
ap2d_vel=cp.ones((ni,nj))*1e-20
su2d=cp.ones((ni,nj))*1e-20
sp2d=cp.ones((ni,nj))*1e-20
ap2d=cp.ones((ni,nj))*1e-20
dudx=cp.ones((ni,nj))*1e-20
dudy=cp.ones((ni,nj))*1e-20
usynt_inlet=cp.ones((nj))*1e-20
vsynt_inlet=cp.ones((nj))*1e-20
wsynt_inlet=cp.ones((nj))*1e-20

# comute Delta_max for LES/DES/PANS models
delta_max=cp.maximum(vol/areas[:,1:],vol/areaw[1:,:])


iter=0



# initialize
u2d,v2d,k2d,om2d,vis2d=modify_init(u2d,v2d,k2d,om2d,vis2d)

# read data from restart
if restart: 
   u2d,v2d,p2d,k2d,om2d,vis2d= read_restart_data()

k2d=cp.maximum(k2d,1e-6)

u2d_face_w,u2d_face_s=compute_face_phi(u2d,u_bc_west,u_bc_east,u_bc_south,u_bc_north,\
    u_bc_west_type,u_bc_east_type,u_bc_south_type,u_bc_north_type)
v2d_face_w,v2d_face_s=compute_face_phi(v2d,v_bc_west,v_bc_east,v_bc_south,v_bc_north,\
    v_bc_west_type,v_bc_east_type,v_bc_south_type,v_bc_north_type)
p2d_face_w,p2d_face_s=compute_face_phi(p2d,p_bc_west,p_bc_east,p_bc_south,p_bc_north,\
    p_bc_west_type,p_bc_east_type,p_bc_south_type,p_bc_north_type)


u_bc_west,v_bc_west,k_bc_west,om_bc_west,u2d_face_w,convw = modify_inlet()

convw,convs=conv(u2d,v2d,p2d_face_w,p2d_face_s)

iter=0

if kom: 
   urf_temp=urfvis # no under-relaxation
   urfvis=1
   vis2d=vist_kom(vis2d,k2d,om2d)
   urfvis=urf_temp

# find max index
#sumax=np.max(su2d.flatten())
#print('[i,j,k]', np.where(su2d == np.amax(su2d)) 

residual_u=0
residual_v=0
residual_p=0
residual_k=0
residual_om=0

######################### start of global iteration process #############################

for iter in range(0,maxit):

      start_time_iter = time.time()
# coefficients for velocities
      start_time = time.time()
# conpute inlet fluc
      if iter == 0:
         u_bc_west,v_bc_west,k_bc_west,om_bc_west,u2d_face_w,convw = modify_inlet()
      aw2d,ae2d,as2d,an2d,su2d,sp2d=coeff(convw,convs,vis2d,1,scheme)

# u2d
# boundary conditions for u2d
      su2d,sp2d=bc(su2d,sp2d,u_bc_west,u_bc_east,u_bc_south,u_bc_north, \
                   u_bc_west_type,u_bc_east_type,u_bc_south_type,u_bc_north_type)
      su2d,sp2d,ap2d=calcu(su2d,sp2d,p2d_face_w,p2d_face_s)

      u2d,residual_u=solve_2d(u2d,aw2d,ae2d,as2d,an2d,su2d,ap2d,convergence_limit_u,nsweep_vel,solver_vel)
      print(f"{'time u: '}{time.time()-start_time:.2e}")

      start_time = time.time()
# v2d
# boundary conditions for v2d
      su2d,sp2d=bc(su2d,sp2d,v_bc_west,v_bc_east,v_bc_south,v_bc_north, \
                   v_bc_west_type,v_bc_east_type,v_bc_south_type,v_bc_north_type)
      su2d,sp2d,ap2d,ap2d_vel=calcv(su2d,sp2d,p2d_face_w,p2d_face_s)
      v2d,residual_v=solve_2d(v2d,aw2d,ae2d,as2d,an2d,su2d,ap2d,convergence_limit_v,nsweep_vel,solver_vel)
      print(f"{'time v: '}{time.time()-start_time:.2e}")

      start_time = time.time()
# pp2d
      convw,convs=conv(u2d,v2d,p2d_face_w,p2d_face_s)
      convw=modify_outlet(convw)
      aw2d,ae2d,as2d,an2d,su2d,ap2d=calcp(pp2d,ap2d_vel)
      pp2d=cp.zeros((ni,nj))
      pp2d,dummy=solve_2d(pp2d,aw2d,ae2d,as2d,an2d,su2d,ap2d,convergence_limit_pp,nsweep_pp,solver_pp)

# correct u, v, w, p
      convw,convs,p2d,u2d,v2d,su2d= correct_u_v_p(u2d,v2d,p2d)
      convw=modify_outlet(convw)

# continuity error
      su2d=convw[0:-1,:]-cp.roll(convw[0:-1,:],-1,axis=0)+convs[:,0:-1]-cp.roll(convs[:,0:-1],-1,axis=1)
      residual_pp=abs(cp.sum(su2d))

      print(f"{'time pp: '}{time.time()-start_time:.2e}")

      u2d_face_w,u2d_face_s=compute_face_phi(u2d,u_bc_west,u_bc_east,u_bc_south,u_bc_north,\
        u_bc_west_type,u_bc_east_type,u_bc_south_type,u_bc_north_type)
      v2d_face_w,v2d_face_s=compute_face_phi(v2d,v_bc_west,v_bc_east,v_bc_south,v_bc_north,\
        v_bc_west_type,v_bc_east_type,v_bc_south_type,v_bc_north_type)
      p2d_face_w,p2d_face_s=compute_face_phi(p2d,p_bc_west,p_bc_east,p_bc_south,p_bc_north,\
        p_bc_west_type,p_bc_east_type,p_bc_south_type,p_bc_north_type)

      start_time = time.time()

      if kom: 

         vis2d=vist_kom(vis2d,k2d,om2d)
# coefficients
         start_time = time.time()
         aw2d,ae2d,as2d,an2d,su2d,sp2d=coeff(convw,convs,vis2d,prand_k,scheme_turb)
# k
# boundary conditions for k2d
         su2d,sp2d=bc(su2d,sp2d,k_bc_west,k_bc_east,k_bc_south,k_bc_north, \
                   k_bc_west_type,k_bc_east_type,k_bc_south_type,k_bc_north_type)
         su2d,sp2d,gen,ap2d=calck(su2d,sp2d,k2d,om2d,vis2d,u2d_face_w,u2d_face_s,v2d_face_w,v2d_face_s)

         k2d,residual_k=solve_2d(k2d,aw2d,ae2d,as2d,an2d,su2d,ap2d,convergence_limit_k,nsweep_kom,solver_turb)
         k2d=np.maximum(k2d,1e-10)
         print(f"{'time k: '}{time.time()-start_time:.2e}")

         start_time = time.time()
# omega
# boundary conditions for om2d
         aw2d,ae2d,as2d,an2d,su2d,sp2d=coeff(convw,convs,vis2d,prand_omega,scheme_turb)
         su2d,sp2d=bc(su2d,sp2d,om_bc_west,om_bc_east,om_bc_south,om_bc_north,\
                   om_bc_west_type,om_bc_east_type,om_bc_south_type,om_bc_north_type)
         su2d,sp2d,ap2d= calcom(su2d,sp2d,om2d,gen)

         aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d=fix_omega()

         om2d,residual_om=solve_2d(om2d,aw2d,ae2d,as2d,an2d,su2d,ap2d,convergence_limit_om,nsweep_kom,solver_turb)
         om2d=cp.maximum(om2d,1e-10)

         print(f"{'time omega: '}{time.time()-start_time:.2e}")

# scale residuals
      residual_u=residual_u/resnorm_vel
      residual_v=residual_v/resnorm_vel
      residual_p=residual_p/resnorm_p
      residual_k=residual_k/resnorm_vel**2
      residual_om=residual_om/resnorm_vel

      resmax=cp.max([residual_u ,residual_v,residual_p])

      print(f"\n{'--iter:'}{iter:d}, {'max residul:'}{resmax:.2e}, {'u:'}{residual_u:.2e}\
, {'v:'}{residual_v:.2e}, {'pp:'}{residual_pp:.2e}, {'k:'}{residual_k:.2e}\
, {'om:'}{residual_om:.2e}\n")

      print(f"\n{'monitor iteration:'}{iter:4d}, {'u:'}{u2d[imon,jmon]: .2e}\
, {'v:'}{v2d[imon,jmon]: .2e}, {'p:'}{p2d[imon,jmon]: .2e}\
, {'k:'}{k2d[imon,jmon]: .2e}, {'om:'}{om2d[imon,jmon]: .2e}\n")



      vismax=cp.max(vis2d.flatten())/viscos
      umax=cp.max(u2d.flatten())
      ommin=cp.min(om2d.flatten())
      kmin=cp.min(k2d.flatten())

      print(f"\n{'---iter: '}{iter:2d}, {'umax: '}{umax:.2e},{'vismax: '}{vismax:.2e}, {'kmin: '}{kmin:.2e}, {'ommin: '}{ommin:.2e}\n")

      print(f"{'time one iteration: '}{time.time()-start_time_iter:.2e}")

      if resmax < sormax: 

         break

######################### end of global iteration process #############################
      
# save data for restart
if save:
   save_data(u2d,v2d,p2d,k2d,om2d,vis2d)

if vtk:
   itstep=ntstep
   save_vtk()

print('program reached normal stop')

