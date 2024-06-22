import numpy as np
from pylab import plot, xlim, ylim, xlabel, ylabel, uniform,randint
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

start_time = time.time()

#-----------------------basis--------

bra0 = np.zeros([1,2])
bra0[0][0]=1

ket0 = np.zeros([2,1])
ket0[0][0]= 1

bra1 = np.zeros([1,2])
bra1[0][1]= 1

ket1 = np.zeros([2,1])
ket1[1][0]= 1

ket00 = np.kron(ket0, ket0)
bra00 = np.kron(bra0, bra0)
ket11 = np.kron(ket1, ket1)
bra11 = np.kron(bra1, bra1)


#----------------identity matrices--------

ii = np.identity(4)
i1 = np.identity(2)


#---------------------quantum cahnnel-------------

def bitflip(state , pa):
    g0a = np.zeros([2,2])
    g1a = np.zeros([2,2])

    g0a[0][0] = np.sqrt(1- (pa/2))
    g0a[1][1] = np.sqrt(1- (pa/2))
    gama0a = np.kron(g0a, i1)

    g1a[0][1] = np.sqrt((pa/2))
    g1a[1][0] = np.sqrt((pa/2))

    gama1a = np.kron(g1a, i1)
    

    x1 = np.dot(np.dot(gama0a , state) , np.transpose(gama0a))
    x2 = np.dot(np.dot(gama1a , state) , np.transpose(gama1a))
    w =  (x1 + x2)
    return w


def phaseflip(state , pa):
    g0a = np.zeros([2,2])
    g1a = np.zeros([2,2])
    
    g0a[0][0] = np.sqrt(1- (pa/2))
    g0a[1][1] = np.sqrt(1- (pa/2))
    gama0a = np.kron(g0a, i1)
    
    g1a[0][0] = np.sqrt((pa/2))
    g1a[1][1] = -np.sqrt((pa/2))
    
    gama1a = np.kron(g1a, i1)
    
    
    x1 = np.dot(np.dot(gama0a , state) , np.transpose(gama0a))
    x2 = np.dot(np.dot(gama1a , state) , np.transpose(gama1a))
    w =  (x1 + x2)
    return w

#--------------------------------------------------------------


sigmay = np.zeros([2,2] , dtype = complex)
sigmay[0][1] = 0-1j
sigmay[1][0] = 0+1j



states=[]
eta = 1  #reflectivity
r1=5#for minimizing discord
r2=5
dmax=0

#-----------------------------------bitflip-----------

l=15 #range for generating states

c1ss = []
c2ss = []
c3ss = []
T1 = 200
T2 = T1  
pa = 0
EOF1 = []



r1=5#for minimizing discord
r2=5
EOF2=[]
c1ss = []
c2ss = []
for i11 in range(-l,l+1):
    for i2 in range(-l,l+1):
        for i3 in range(-l,l+1):
    

            c1 = i11/l
            c2 = i2/l
            c3 = i3/l
            if (1-c1-c2-c3)>=0 and (1-c1+c2+c3)>=0 and (1+c1-c2+c3)>=0 and (1+c1+c2-c3)>=0:
                bb= np.zeros([4,4])
                bb[0][0] = 1 + c3
                bb[0][3] = c1 - c2
                bb[3][0] = c1 - c2
                bb[1][1] = 1 - c3
                bb[1][2] = c1 + c2
                bb[2][1] = c1 + c2
                bb[2][2] = 1 - c3
                bb[3][3] = 1 + c3
                
                
                
                c1ss.append(c1)
                c2ss.append(c2)
                c3ss.append(c3)
                bb = 0.25*bb
                
                bb = (1-eta)*ii/4 + eta*bb
                
                rhotild = np.dot(np.dot(np.kron(sigmay, sigmay) ,np.conjugate(bitflip(bb, pa))) , np.kron(sigmay, sigmay))
                
                e1 , e2 = np.linalg.eig(np.dot(bitflip(bb, pa),rhotild))
                
                e1.sort()
                
                C = max(0 , np.sqrt(e1[3]) -np.sqrt( e1[2]) - np.sqrt(e1[1]) - np.sqrt(e1[0]))
                
                if ((1 - np.sqrt( 1 - C**2 ))/2)>10**-15:
                    EOF2.append((-((1 + np.sqrt( 1 - C**2 ))/2) * math.log2((1 + np.sqrt( 1 - C**2 ))/2)) 
                                -((1 - np.sqrt( 1 - C**2 ))/2) * math.log2((1 - np.sqrt( 1 - C**2 ))/2))
                else:
                    EOF2.append((-((1 + np.sqrt( 1 - C**2 ))/2) * math.log2((1 + np.sqrt( 1 - C**2 ))/2)))#this is 0
                    
                    
                    
print(len(c1ss))                   
                    
eof = np.zeros(len(c1ss))           
c1s = np.zeros(len(c1ss))           
c2s = np.zeros(len(c1ss))  
c3s = np.zeros(len(c1ss))           
         
for i in range(len(c1ss)):
    c1s[i] = c1ss[i] #- c1ss2[i] 
    c2s[i] = c2ss[i] #- c2ss2[i] 
    c3s[i] = c3ss[i]
    eof[i] = EOF2[i] #- discordss1[i]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
img = ax.scatter(c1s , c2s , c3s , c=eof , cmap=plt.hot() , s=50 , vmin = 0 , vmax=1 )
fig.colorbar(img)

ax.set(xlabel='C1', ylabel='C2', zlabel='c3')
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:",execution_time)
