import numpy
import csv

from matplotlib import pyplot, cm
nx = 41
ny = 41

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
X, Y = numpy.meshgrid(x, y)

u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx))

def read_data(path,n,u,v,p):
    with open(path+'u_data_'+str(n)+'.csv', 'r') as f:
        reader = csv.reader(f)
        index=-1
        for row in reader:
            index=index+1
            for cal in range(41):
                u[index,cal]=float(row[cal]);
    with open(path+'v_data_'+str(n)+'.csv', 'r')  as f:
        reader = csv.reader(f)
        index=-1
        for row in reader:
            index=index+1
            for cal in range(41):
                v[index,cal]=float(row[cal]);
    with open(path+'p_data_'+str(n)+'.csv', 'r')  as f:
        reader = csv.reader(f)
        index=-1
        for row in reader:
            index=index+1
            for cal in range(41):
                p[index,cal]=float(row[cal]);

path='./Data/'
n=100
read_data(path,n,u,v,p)
fig = pyplot.figure(figsize=(11,7), dpi=100)
# plotting the pressure field as a contour
pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
# plotting the pressure field outlines
pyplot.contour(X, Y, p, cmap=cm.viridis)
# plotting velocity field
pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
pyplot.xlabel('X')
pyplot.ylabel('Y');

n=700
read_data(path,n,u,v,p)
fig = pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X, Y, p, cmap=cm.viridis)
pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
pyplot.xlabel('X')
pyplot.ylabel('Y');

fig = pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X, Y, p, cmap=cm.viridis)
pyplot.streamplot(X, Y, u, v)
pyplot.xlabel('X')
pyplot.ylabel('Y');

pyplot.show()


