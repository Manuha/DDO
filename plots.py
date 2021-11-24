# importing libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
# import scipy.ndimage.interpolation as ip_plot
import scipy.interpolate as ip_plot
from scipy.interpolate import griddata
from numpy import genfromtxt

def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
rng = np.random.default_rng()

points = rng.random((1000, 2))

values = func(points[:,0], points[:,1])

# print(type(points))
# print(values.shape)

# Italy
ω_file='data/italy_ω.csv'
α_file='data/italy_α.csv'
error_file='data/italy_error.csv'

# # USA
# ω_file='data/usa_ω.csv'
# α_file='data/usa_α.csv'
# error_file='data/usa_error.csv'

omega = genfromtxt(ω_file, delimiter=',')
alpha = genfromtxt(α_file, delimiter=',')
error = genfromtxt(error_file, delimiter=',')

X = omega.reshape(-1,1)
Y = alpha.reshape(-1,1)
Z = error.reshape(-1,1)
print(alpha.min(),alpha.max())
# xi = np.linspace(omega.min(),omega.max(),1000)
# yi = np.linspace(alpha.min(),alpha.max(),1000)
grid_x, grid_y = np.mgrid[omega.min():omega.max():2000j, alpha.min():alpha.max():2000j]
# print("ab")
points = np.concatenate((X,Y), axis=1)
print(grid_x[:,0])
print(Z.shape)
# zi = griddata((omega,alpha), error, (xi[None,:], yi[:,None]), method='cubic')
zi = griddata( points, Z, (grid_x,grid_y), method='cubic',fill_value=0).reshape(2000,2000)
print(zi.shape)

# xig, yig = np.meshgrid(grid_x,grid_y)

ax = plt.axes(projection ='3d')
fig = plt.figure()
ax.set_xlabel("Omega")
ax.set_ylabel("Alpha")
ax.set_zlabel("Error")
surf = ax.plot_surface(grid_x,grid_y, zi, cmap='gist_earth')
# surf = ax.plot_surface(omega,alpha,error, cmap='gist_earth')
fig.colorbar(surf, shrink=0.5, aspect=5)

 
# syntax for 3-D plotting
 
# syntax for plotting
# ax.plot_surface(ω, α, error, cmap ='viridis', edgecolor ='green')
# a = np.array(ω, α, error)
# ip_plot.bisplrep(ω, α, error,s=1000000)
# ax.set_title('Surface plot geeks for geeks')

plt.show()