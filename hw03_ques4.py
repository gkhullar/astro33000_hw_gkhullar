import numpy as np
from mpi4py import MPI as mpi
import string
import sys
import time
import matplotlib.pyplot as plt
from time import clock
import cPickle

comm = mpi.COMM_WORLD
mpi_rank, mpi_size = comm.Get_rank(), comm.Get_size()

###################################################

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


m0 = 1.989e30 # 1 M_solar(in kilograms)
t0 = 86400 #1 day (in seconds)
r0 = 1.496e+11 #1 AU (in metres)

N=10;D=3
epsilon =1e-5

x=np.zeros((N,D))
v=np.zeros((N,D))

name = np.loadtxt('planets.dat', usecols=[0], unpack=True, dtype=str)
mp, xp, yp, zp, vxp, vyp, vzp = np.loadtxt('planets.dat', usecols=(1,2,3,4,5,6,7), unpack=True)
#print name, mp, xp, yp, zp, vxp, vyp, vzp


x[:,0]=xp;x[:,1]=yp;x[:,2]=zp   #AU
v[:,0]=vxp;v[:,1]=vyp;v[:,2]=vzp #AU/day
    
print 'v0 is', r0/t0
#x *= 0.01 #cm to metres
#x /= r0

#v *= 0.01 #cm/s to m/s
#v /= (r0/t0)

m = np.copy(mp)
#m *=0.001 # grams to kilograms
#m /= m0    #Solar masses

G = 6.67e-11 #SI units
G /= r0**3/(m0*(t0**2))
print G

t_start=0
t_end = 1000*365  # in days (t0) units
dt=5    #5 days per integration


timearray = np.arange(t_start,t_end,dt)

t=0
x1 =np.copy(x)
v1 = np.copy(v)


tstart = clock()
#time,position,velocity = leapfrog_mpiversion(accvec2, x0, v0,t_start, t_end, dt, G, m, epsilon)
##################################
    
if mpi_rank==0: 
    xvec = []; vvec=[]

h = N/mpi_size

while t < t_end: #Looping over time steps
    
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
    comm.Barrier()

    if mpi_rank==0:
        if t%(10) == 0.:
            xvec.append(np.copy(x1))
            vvec.append(np.copy(v1))
        
    t += dt


if mpi_rank==0:
    print "accvec2 completed in %.2f seconds"%(clock()-tstart)
    xout = np.asarray(xvec)
    vout = np.asarray(vvec)
    



# Plotting routine
#
#c = ['slateblue', 'dimgrey', 'green', 'red','yellow']
    c = ['slateblue', 'red', 'green','dimgrey','yellow','magenta','blue','black','brown','cyan']
#plt.xscale('log'); plt.yscale('log')
#plt.xlim(t0+1.e-6,tf); plt.ylim(y0+1.e-6,3.)
    #plt.figure(figsize=(3,3))
    #plt.xlabel('$x$'); plt.ylabel('$y$')
    #for i in range(N):
    #    plt.plot(xout[:,i,0], xout[:,i,1] , lw=0.15, c=c[i], alpha=1., label='part. %s'%(i+1))
    #plt.grid()
    #plt.legend(frameon=False, loc='lower left', fontsize=5)   
    #plt.show()
    #plt.savefig('orbits_hw3_ques4.pdf')

    f1 = [xout,vout]
    print f1
    f = open('orbitsdata_q4.dat','wb')
    cPickle.dump(f1,f)

