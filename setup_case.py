import cupy as cp

def setup_case():
   ########### section 1 choice of differencing scheme ###########
   scheme = 'h'  # hybrid
   scheme_turb = 'h'  # hybrid upwind-central 

   ########### section 2 turbulence models ###########
   cmu = 0.09
   kom = True
   c_omega_1 = 5. / 9.
   c_omega_2 = 3. / 40.
   prand_omega = 2.0
   prand_k = 2.0

   ########### section 3 restart/save ###########
   restart = False
   save = True

   ########### section 4 fluid properties ###########
   viscos = 1 / 5200

   ########### section 5 relaxation factors ###########
   urfvis = 0.5
   urf_vel = 0.5
   urf_k = 0.5
   urf_p = 1.0
   urf_omega = 0.5

   ########### section 6 number of iteration and convergence criteria ###########
   maxit = 20000
   min_iter = 1
   sormax = 1e-7

   solver_vel = 'direct'
   solver_pp = 'direct'
   solver_turb = 'direct'

   nsweep_vel = 50
   nsweep_pp = 50
   nsweep_kom = 50
   convergence_limit_u = 1e-6
   convergence_limit_v = 1e-6
   convergence_limit_k = 1e-6
   convergence_limit_om = 1e-6
   convergence_limit_pp = 5e-4

   ########### section 7 all variables are printed during the iteration at node ###########
   imon = 0
   jmon = 10

   ########### section 8 save data for post-processing ###########
   vtk = False
   save_all_files = False
   vtk_file_name = 'bound'

   ########### section 9 residual scaling parameters ###########
   uin = 20
   resnorm_p = uin * y2d[1, -1]
   resnorm_vel = uin**2 * y2d[1, -1]

   ########### Section 10 boundary conditions ###########

   # boundary conditions for u
   u_bc_west = cp.ones(nj)
   u_bc_east = cp.zeros(nj)
   u_bc_south = cp.zeros(ni)
   u_bc_north = cp.zeros(ni)

   u_bc_west_type = 'n' 
   u_bc_east_type = 'n' 
   u_bc_south_type = 'd'
   u_bc_north_type = 'd'

   # boundary conditions for v
   v_bc_west = cp.zeros(nj)
   v_bc_east = cp.zeros(nj)
   v_bc_south = cp.zeros(ni)
   v_bc_north = cp.zeros(ni)

   v_bc_west_type = 'n' 
   v_bc_east_type = 'n' 
   v_bc_south_type = 'd'
   v_bc_north_type = 'd'

   # boundary conditions for p
   p_bc_west = cp.zeros(nj)
   p_bc_east = cp.zeros(nj)
   p_bc_south = cp.zeros(ni)
   p_bc_north = cp.zeros(ni)

   p_bc_west_type = 'n'
   p_bc_east_type = 'n'
   p_bc_south_type = 'n'
   p_bc_north_type = 'n'

   # boundary conditions for k
   k_bc_west = cp.zeros(nj)
   k_bc_east = cp.zeros(nj)
   k_bc_south = cp.zeros(ni)
   k_bc_north = cp.zeros(ni)

   k_bc_west_type = 'n'
   k_bc_east_type = 'n'
   k_bc_south_type = 'd'
   k_bc_north_type = 'd'

   # boundary conditions for omega
   om_bc_west = cp.zeros(nj)
   om_bc_east = cp.zeros(nj)

   # Compute south and north wall distances for omega BCs using CuPy
   xwall_s = 0.5 * (x2d[0:-1, 0] + x2d[1:, 0])
   ywall_s = 0.5 * (y2d[0:-1, 0] + y2d[1:, 0])
   dist2_s = (yp2d[:, 0] - ywall_s)**2 + (xp2d[:, 0] - xwall_s)**2
   om_bc_south = 10 * 6 * viscos / 0.075 / dist2_s

   xwall_n = 0.5 * (x2d[0:-1, -1] + x2d[1:, -1])
   ywall_n = 0.5 * (y2d[0:-1, -1] + y2d[1:, -1])
   dist2_n = (yp2d[:, -1] - ywall_n)**2 + (xp2d[:, -1] - xwall_n)**2
   om_bc_north = 10 * 6 * viscos / 0.075 / dist2_n

   om_bc_west_type = 'n'
   om_bc_east_type = 'n'
   om_bc_south_type = 'd'
   om_bc_north_type = 'd'

   return 
