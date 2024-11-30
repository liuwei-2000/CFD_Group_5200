import cupy as cp
import sys

ni = 1
nj = 96
yfac = 1.15  # stretching
viscos = 1 / 5200
dy = 0.1
ymax = 2
xmax = 0.6

# Create CuPy array for yc, initialized to zeros
yc = cp.zeros(nj + 1)
yc[0] = 0.0

# Loop to generate yc values with stretching
for j in range(1, int(nj / 2) + 1):
    yc[j] = yc[j - 1] + dy
    dy = yfac * dy

ymax_scale = yc[int(nj / 2)]

# Cell faces
for j in range(1, int(nj / 2) + 1):
    yc[j] = yc[j] / ymax_scale
    yc[nj - j + 1] = ymax - yc[j - 1]

yc[int(nj / 2)] = 1

print('y+', 0.5 * yc[1] / viscos)

# Create 2D y2d array by repeating yc across rows (same as cp.repeat)
y2d = cp.repeat(yc[None, :], repeats=ni + 1, axis=0)

# Append nj to y2d
y2d = cp.append(y2d, nj)

# Save to file (CuPy works similarly to NumPy for saving)
cp.savetxt('y2d.dat', y2d)

# x grid using linspace in CuPy
xc = cp.linspace(0, xmax, ni + 1)

# Create 2D x2d array by repeating xc across columns
x2d = cp.repeat(xc[:, None], repeats=nj + 1, axis=1)

# Append ni to x2d
x2d = cp.append(x2d, ni)

# Save to file
cp.savetxt('x2d.dat', x2d)

# Check it - loading data using CuPy instead of NumPy
datay = cp.loadtxt("y2d.dat")
y = datay[0:-1]
nj = int(datay[-1])

y2 = cp.zeros((ni + 1, nj + 1))
y2 = cp.reshape(y, (ni + 1, nj + 1))

datax = cp.loadtxt("x2d.dat")
x = datax[0:-1]
ni = int(datax[-1])

x2 = cp.zeros((ni + 1, nj + 1))
x2 = cp.reshape(x, (ni + 1, nj + 1))
