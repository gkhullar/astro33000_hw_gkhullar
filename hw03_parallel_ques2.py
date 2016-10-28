

import numpy as np
from mpi4py import MPI as mpi
import string
import sys
import time
import matplotlib.pyplot as plt
from time import clock


comm = mpi.COMM_WORLD
mpi_rank, mpi_size = comm.Get_rank(), comm.Get_size()

###################################################

# number of dimensions, number of particles
D = 3; N = 50; 

inn, jnn = np.indices( (N,N) )            # for the dr array
inn3, jnn3, knn3 = np.indices( (N,N,D) )  # for the acc array

#def accvec1(x, mp):
#    """ vectorized version of acceleration calculation"""
#    dx = x[jnn,:] - x[inn,:]    
#    # compute 1/r^3 adding unity at diagonal to avoid division by zero
#    ir3 = np.power((np.sum(dx*dx,2) + np.eye(N,N)), -1.5)  
#   # ve
#    acc = np.sum(mp[jnn3] * dx[inn3,jnn3,knn3] * ir3[inn3,jnn3], axis=1)
#    return acc


def remove_i(x, i):
    """Drops the ith element of an array."""
    shape = (x.shape[0]-1,) + x.shape[1:]
    y = np.empty(shape, dtype=float)
    y[:i] = x[:i]
    y[i:] = x[i+1:]
    return y

def accvec2(i, x, G, m, epsilon):  #Added epsilon softening now
    """The acelleration of the ith mass."""
    x_i = x[i]
    x_j = remove_i(x, i)
    m_j = remove_i(m, i)
    
    #diff = x_j - x_i
    diff = x_j - x_i + epsilon
    
    mag3 = np.sum(diff**2, axis=1)**1.5
    result = G * np.sum(diff * (m_j / mag3)[:,np.newaxis], axis=0)
    return result


####################################################

#def leapfrog_mpi(x0, v0, m, tstep, Tf, D=3, tunit = 3.154e7, Munit = 1.988e30, runit = 1.496e11,epsilon):
def leapfrog_mpi(x0, v0, Tf, tstep, G, m, epsilon):

    N = m.size
    T = int(Tf/tstep)
    
    #G0 = 6.674e-11
    #G = G0 * Munit * tunit**2 / runit**3

    xs = np.zeros((T,N,D)); vs = np.zeros((T,N,D))
    xcur = np.copy(x0)
    vcur = np.copy(v0)
    for i in range(N-1):
        vcur += tstep/2 * accvec2(i, xcur, G, m,epsilon)
    ts = np.arange(0, Tf, tstep)

    h = N / mpi_size
        
    for t in range(0, T):
        xcur += tstep*vcur
        localv = np.zeros((h,3))
        comm.Scatter(vcur, localv, root=0)
        for i in range(h*mpi_rank-1, h*mpi_rank+h-1):
            acc = accvec2(i, xcur, G, m,epsilon)
            localv[i] += tstep*acc    #Step Nora and I decided was wrong
        comm.Gather(localv, vcur, root=0)
        xs[t] = xcur
        vs[t] = vcur
            
    return ts, xs, vs

###################################################
#Initial Conditions

m0 = 1.989e30 #kg
t0 = 3.154e+7 #1 year (in seconds)
r0 = 1.496e+11 #1 AU (in meters)

epsilon =1e-6

x0 = np.random.rand(N, D)
v0 = np.zeros((N, D), dtype=float)
v0 /= (r0/t0)

m = np.ones(N, dtype=float)

G = 6.67e-11 #SI units
G /= r0**3/(m0*(t0**2))   # G_tilda = G/(r0^3/m0.t0^2)
#print G
#print m
#print v0
#print x0

t_start=0
t_end = 2  # in Years (t0) units
dt=1e-3


tstart = clock()
time,position,velocity = leapfrog_mpi(x0, v0, t_end, dt, G, m, epsilon)
print "accvec2 completed in %.2f seconds"%(clock()-tstart)


#
# Plotting routine
#

#c = ['slateblue', 'dimgrey', 'green', 'red','yellow']

plot_pretty()
#plt.xscale('log'); plt.yscale('log')
#plt.xlim(t0+1.e-6,tf); plt.ylim(y0+1.e-6,3.)
plt.figure(figsize=(3,3))
plt.xlabel('$x$'); plt.ylabel('$y$')

for i in range(N):
    plt.plot(position[:,i,0], position[:,i,1] , lw=0.5, c='slateblue', alpha=1., label='part. %s'%(i+1))
plt.xlim(-2,2); plt.ylim(-2,2)
plt.grid()
plt.legend(frameon=False, loc='lower left', fontsize=5)   
#plt.show()
plt.savefig('orbits_hw3.jpg')


