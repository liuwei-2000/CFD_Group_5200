import cupy as cp
import scipy.io as sio
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython import display

plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.max_open_warning': 0})

plt.interactive(True)

viscos = 1 / 5200

# makes sure figures are updated when using ipython
display.clear_output(wait=True)

# Load data using CuPy
datax = cp.loadtxt("x2d.dat")
x = datax[0:-1]
ni = int(datax[-1])
datay = cp.loadtxt("y2d.dat")
y = datay[0:-1]
nj = int(datay[-1])

# Create CuPy arrays for x2d and y2d
x2d = cp.zeros((ni + 1, nj + 1))
y2d = cp.zeros((ni + 1, nj + 1))

x2d = cp.reshape(x, (ni + 1, nj + 1))
y2d = cp.reshape(y, (ni + 1, nj + 1))

# Compute cell centers using CuPy
xp2d = 0.25 * (x2d[0:-1, 0:-1] + x2d[0:-1, 1:] + x2d[1:, 0:-1] + x2d[1:, 1:])
yp2d = 0.25 * (y2d[0:-1, 0:-1] + y2d[0:-1, 1:] + y2d[1:, 0:-1] + y2d[1:, 1:])

y = yp2d[0, :]

# Load data arrays as CuPy arrays
u2d = cp.load('u2d_saved.cpy')
p2d = cp.load('p2d_saved.cpy')
v2d = cp.load('v2d_saved.cpy')
k2d = cp.load('k2d_saved.cpy')
om2d = cp.load('om2d_saved.cpy')
vis2d = cp.load('vis2d_saved.cpy')

# Average in x direction
u = cp.mean(u2d, axis=0)
v = cp.mean(v2d, axis=0)
k = cp.mean(k2d, axis=0)
om = cp.mean(om2d, axis=0)
vis = cp.mean(vis2d, axis=0)
eps = 0.09 * k * om

# Compute gradient in y direction
dudy = cp.gradient(u, y)
uv = -(vis - viscos) * dudy

# Save data to text files using CuPy
cp.savetxt('y_u_k_eps_uv_5200-RANS-code.txt', cp.c_[y, u, k, eps, uv])
cp.savetxt('y_u_k_om_uv_5200-RANS-code.txt', cp.c_[y, u, k, om, uv])

ustar = (viscos * u[0] / y[0]) ** 0.5
yplus = y * ustar / viscos

# Load DNS data using CuPy
DNS_mean = cp.genfromtxt("LM_Channel_5200_mean_prof.dat", comments="%")
y_DNS = DNS_mean[:, 0]
yplus_DNS = DNS_mean[:, 1]
u_DNS = DNS_mean[:, 2]

DNS_stress = cp.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat", comments="%")
u2_DNS = DNS_stress[:, 2]
v2_DNS = DNS_stress[:, 3]
w2_DNS = DNS_stress[:, 4]
uv_DNS = DNS_stress[:, 5]

k_DNS = 0.5 * (u2_DNS + v2_DNS + w2_DNS)

# Find equidistant DNS cells in log-scale
xx = 0.
jDNS = [1] * 40
for i in range(0, 40):
    i1 = (cp.abs(10. ** xx - yplus_DNS)).argmin()
    jDNS[i] = int(i1)
    xx = xx + 0.2

# Plotting using Matplotlib
########################################## U 
fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
plt.semilogx(yplus, u, 'b-')
plt.semilogx(yplus_DNS[jDNS], u_DNS[jDNS], 'bo')
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")
plt.axis([1, 5200, 0, 28])
plt.savefig('u_log_5200-channel.png', bbox_inches='tight')

########################################## uv 
fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
vist = vis - viscos
dudy = cp.gradient(u, y)
uv = -vist * dudy
plt.plot(yplus, uv, 'b-')
plt.plot(yplus_DNS[jDNS], uv_DNS[jDNS], 'bo')
plt.ylabel(r"$\overline{u'v'}$")
plt.xlabel("$y^+$")
plt.axis([1, 10400, -1, 1])
plt.savefig('uv_5200-channel.png', bbox_inches='tight')

########################################## k 
fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
plt.plot(yplus, k, 'b-')
plt.ylabel(r"$k^+$")
plt.xlabel("$y^+$")
plt.axis([1, 10400, 0, 5])
plt.savefig('k_5200-channel.png', bbox_inches='tight')

########################################## vis 
fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
plt.plot(yplus, vis / viscos, 'b-')
plt.ylabel(r"$\nu_t/\nu$")
plt.xlabel("$y^+$")
plt.axis([1, 10400, 0, 700])
plt.savefig('vis_5200-channel.png', bbox_inches='tight')
