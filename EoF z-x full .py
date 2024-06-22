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

#-----------------bell state--------

bell1 = (1/(np.sqrt(2))) * (ket00 + ket11)
bell2 = (1/(np.sqrt(2))) * (bra00 + bra11)

belld = np.kron(bell1, bell2)

#----------------identity matrices--------

ii = np.identity(4)
i1 = np.identity(2)

#-----------------------------entropy function--------------------

def entropy( s):
    ent = 0
    ev , es = np.linalg.eig(s)
    for i in ev:
        if i>0:
            ent += -i.real*math.log2(i.real)
    return ent

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
ff = 0.005   #decoherence rate
eta = 1  #reflectivity
p0 = 0.5 #probability of presence of the object
p1 = 1 - p0
r1=5#for minimizing discord
r2=5
dmax=0

#-----------------------------------bitflip-----------

l=13 #range for generating states

c1ss = []
c2ss = []
q = 0 #counter
c3 = 0.6
T1 = 200
T2 = T1  
pa = 0.4
EOF1 = []
for i11 in range(-l,l+1):
    for i2 in range(-l,l+1):
            

            c1  = i11/l
            c2 = i2/l
            
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
                
                
                bb = 0.25*bb
                c1ss.append(c1)
                c2ss.append(c2)
                
                bb = (1-eta)*ii/4 + eta*bb
                
                rhotild = np.dot(np.dot(np.kron(sigmay, sigmay) ,np.conjugate(phaseflip(bb, pa))) , np.kron(sigmay, sigmay))
                
                e1 , e2 = np.linalg.eig(np.dot(phaseflip(bb, pa),rhotild))
                
                e1.sort()
                
                C = max(0 , np.sqrt(e1[3]) -np.sqrt( e1[2]) - np.sqrt(e1[1]) - np.sqrt(e1[0]))
                
                
                
                if ((1 - np.sqrt( 1 - C**2 ))/2)>0:
                    EOF1.append((-((1 + np.sqrt( 1 - C**2 ))/2) * math.log2((1 + np.sqrt( 1 - C**2 ))/2)) 
                                -((1 - np.sqrt( 1 - C**2 ))/2) * math.log2((1 - np.sqrt( 1 - C**2 ))/2))
                else:# else, the second term is zero
                    EOF1.append((-((1 + np.sqrt( 1 - C**2 ))/2) * math.log2((1 + np.sqrt( 1 - C**2 ))/2)))
                           
                    
                    

ax = plt.figure().add_subplot(projection='3d')



r1=5#for minimizing discord
r2=5
EOF2=[]
c1ss = []
c2ss = []
for i11 in range(-l,l+1):
    for i2 in range(-l,l+1):
            

            c1  = i11/l
            c2 = i2/l
            
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
                
                
                bb = 0.25*bb
                c1ss.append(c1)
                c2ss.append(c2)
                
                bb = (1-eta)*ii/4 + eta*bb
                
                rhotild = np.dot(np.dot(np.kron(sigmay, sigmay) ,np.conjugate(bitflip(bb, pa))) , np.kron(sigmay, sigmay))
                
                e1 , e2 = np.linalg.eig(np.dot(bitflip(bb, pa),rhotild))
                
                e1.sort()
                
                C = max(0 , np.sqrt(e1[3]) -np.sqrt( e1[2]) - np.sqrt(e1[1]) - np.sqrt(e1[0]))
                
                if ((1 - np.sqrt( 1 - C**2 ))/2)>0:
                    EOF2.append((-((1 + np.sqrt( 1 - C**2 ))/2) * math.log2((1 + np.sqrt( 1 - C**2 ))/2)) 
                                -((1 - np.sqrt( 1 - C**2 ))/2) * math.log2((1 - np.sqrt( 1 - C**2 ))/2))
                else:
                    EOF2.append((-((1 + np.sqrt( 1 - C**2 ))/2) * math.log2((1 + np.sqrt( 1 - C**2 ))/2)))
                    
                    
                    
print(len(c1ss))                   
                    
eof2 = np.zeros(len(EOF2))        
c1s = np.zeros(len(c1ss))           
c2s = np.zeros(len(c1ss))           
for i in range(len(eof2)):
    c1s[i] = c1ss[i] 
    c2s[i] = c2ss[i]
    eof2[i] = EOF2[i] - EOF1[i]           

        
#ax = plt.figure().add_subplot(projection='3d')

ax.plot_trisurf(c1s , c2s , eof2 , edgecolor='blue' , alpha = 0.02)


ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-0.1,0.1)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:",execution_time)
