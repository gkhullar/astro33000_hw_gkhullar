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
#D = 3; N = 20; 

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

###################################

#Initial Conditions

N=2; D=3

######

m0 = 1.989e30 #kg
t0 = 3.154e+7 #1 year (in seconds)
r0 = 1.496e+11 #1 AU (in meters)

x0 = np.zeros((N,D))
x0[0,0]=1; #x=r0,y=0,z=0
v0 = np.zeros((N,D))
v0[0,1]=2*np.pi
m = np.array([3.0034896149157645e-6,1])

G = 6.67e-11 #SI units
G /= r0**3/(m0*(t0**2))   # G_tilda = G/(r0^3/m0.t0^2)
print G

epsilon =1e-6

#x0 = np.random.rand(N, D)
#v0 = np.zeros((N, D), dtype=float)
#v0 /= (r0/t0)

#m = np.ones(N, dtype=float)

t_start=0
t_end = 1  # in Years (t0) units
dt=1e-3

timearray = np.arange(t_start,t_end,dt)

t=0
x1 =np.copy(x0)
v1 = np.copy(v0)
print x1
print v1

tstart = clock()
#time,position,velocity = leapfrog_mpiversion(accvec2, x0, v0,t_start, t_end, dt, G, m, epsilon)
##################################
    
if mpi_rank==0: 
    xvec = []; vvec=[]

h = N/mpi_size

while t < t_end: #Looping over time steps
    comm.Barrier()

    localv = np.zeros((h,D))
    localx = np.zeros((h,D))
    comm.Scatter(v1, localv, root=0)
    comm.Scatter(x1, localx, root=0)
     
    N_val = np.arange(mpi_rank*h, (mpi_rank+1)*h, dtype=int)
    
    for i,n in enumerate(N_val):    #Looping over number of particles per node
        #The code below gives integer steps to velocity too
        f_old = accvec2(n,x1,G,m,epsilon)
        localx[i] += dt*localv[i] + 0.5*f_old*(dt**2)
        x1[n] = localx[i]
        localv[i] += 0.5*dt*(accvec2(n,x1,G,m,epsilon) + f_old)
        
    x1 = np.zeros((N,D),dtype=float)
    v1 = np.zeros((N,D),dtype=float)

    comm.Allgather(localv, v1)
    comm.Allgather(localx, x1)
    

    if mpi_rank==0:
        xvec.append(np.copy(x1))
        vvec.append(np.copy(v1))
        
    t += dt


if mpi_rank==0:
    print "accvec2 completed in %.2f seconds"%(clock()-tstart)
    xout = np.asarray(xvec)
    vout = np.asarray(vvec)
    print 'position array is', xout
    print 'velocity array is', vout
# Plotting routine
#
#c = ['slateblue', 'dimgrey', 'green', 'red','yellow']
#plt.xscale('log'); plt.yscale('log')
#plt.xlim(t0+1.e-6,tf); plt.ylim(y0+1.e-6,3.)
    plt.figure(figsize=(3,3))
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.plot(xout[:,0,0], xout[:,0,1] , lw=1.5, c='slateblue', alpha=1., label='earth')
    #plt.plot(xout[0,1,0], xout[0,1,1],'*', lw=0.5, c='yellow', alpha=1., label='particles')
    #plt.xlim(-2,2); plt.ylim(-2,2)
    plt.grid()
    plt.legend(frameon=False, loc='lower left', fontsize=5)   
    #plt.show()
    plt.savefig('orbits_hw3.pdf')