import numpy as np
from pylab import plot, xlim, ylim, xlabel, ylabel, uniform,randint
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

start_time = time.time()

#-----------------------basis--------

bra0 = np.zeros([1,2],dtype = complex)
bra0[0][0]=1

ket0 = np.zeros([2,1],dtype = complex)
ket0[0][0]= 1

bra1 = np.zeros([1,2],dtype = complex)
bra1[0][1]= 1

ket1 = np.zeros([2,1],dtype = complex)
ket1[1][0]= 1

ket00 = np.kron(ket0, ket0)
bra00 = np.kron(bra0, bra0)
ket11 = np.kron(ket1, ket1)
bra11 = np.kron(bra1, bra1)


#----------------identity matrices--------

ii = np.identity(4)
i1 = np.identity(2)

#-----------------------------entropy function--------------------

def entropy( s):
    ent = 0
    ev , es = np.linalg.eig(s)
    for i in ev:
        if i>10**(-15):
            ent += -i*math.log2(i)
    return ent

#-------------------------partial trace function----------------

def ptrace(y):
    x = np.zeros([2,2],dtype = complex)
    x[0][0] = y[0][0] + y[1][1] 
    x[0][1] = y[0][2] + y[1][3]
    x[1][0] = y[2][0] + y[3][1]
    x[1][1] = y[2][2] + y[3][3]
    return x

#--------------------------measurement function--------------------

def povm( pvm1 , state):
    mes =  np.matmul(np.matmul(pvm1, state),np.transpose(np.conjugate(pvm1)))
    
    rhoc = ptrace(mes)/np.trace(np.matmul(np.matmul(np.transpose(np.conjugate(pvm1)),pvm1), state))
    return rhoc

#------------------------probability of measurement outcome---------------
def pmeas(pvm,state):
    pi = np.trace(np.dot(np.dot(np.transpose(np.conjugate(pvm)),pvm), state))
    return pi


#--------------------------------------------------------------

eta = 0.5  #reflectivity
p0 = 0.5 #probability of presence of the object
p1 = 1 - p0
r1=5#for minimizing discord
r2=5
dmax=0

#--------------------------------------------
pa = 0
deltabarr = []
l=10 #range for generating states
discordss1 =[]
c1ss = []
c2ss = []
c3ss=[]
for i11 in range(-l,l+1):
    for i2 in range(-l,l+1):
        for i3 in range(-l,l+1):
            

            c1  = i11/l
            c2 = i2/l
            c3 = i3/l
            if (1-c1-c2-c3)>=0 and (1-c1+c2+c3)>=0 and (1+c1-c2+c3)>=0 and (1+c1+c2-c3)>=0:
                bb= np.zeros([4,4],dtype = complex)
                bb[0][0] = 1 + c3
                bb[0][3] = c1 - c2
                bb[3][0] = c1 - c2
                bb[1][1] = 1 - c3
                bb[1][2] = c1 + c2
                bb[2][1] = c1 + c2
                bb[2][2] = 1 - c3
                bb[3][3] = 1 + c3
                
                
                #states.append( 1/4 * bb)
                    
#-----------------------computing discord of encoding-------------
              
                bell =0.25*bb
                
                   
              #  w = bitflip(bell, pa)
                   
                #----------------------------
                rhob = ptrace(bell)
                
                
                ###################################### signal returns
                #######################################

                
                w2 = (1-eta)*ii/4 + eta*bell
                
               # w2 =  bitflip(w1, pa)
                   
                #---------------sub state----------- 
                    
                rhob2 = ptrace(w2)
                 
                
                theta = 0
                phi = 0
                dd11=0
                dd22=1
                r1,r2=5,5
                for t1 in range(31):
                    for t2 in range(31):
                        theta = t1/10
                        phi = t2/10
                #------------------------------------------------------------------------------------         
                        zero0 = (np.cos(theta))*ket0 + (np.exp(1j*phi))*(np.sin(theta))*ket1
                        zero1 = np.transpose(np.conjugate(zero0))
                        
                        povm1 = np.kron( np.identity(2) , np.outer(zero0, zero1))
                
                        one0 = np.exp((-1j)*phi)*np.sin(theta)*ket0 - np.cos(theta)*ket1 
                        one1 = np.transpose(np.conjugate(one0))
                        
                        povm2 = np.kron( np.identity(2) , np.outer(one0 , one1))
                        
                #----------------conditional state------------
                
                        rhoc21 = povm(povm1, w2)
                        rhoc22 = povm(povm2, w2)
                        rhoc2 = pmeas(povm1, w2)*rhoc21 + pmeas(povm1, w2)*rhoc22
                #--------------------average state----------
                        
                        rhobar = p0 *w2 + p1*ii/4
                        
                #-------------------measurement on average state----
                
                        rhobarc1 = p0*rhoc21 + p1*i1/2
                        rhobarc2 = p0*rhoc22 + p1*i1/2
                
                #-----------average substate-----------
                
                        rhobarb = ptrace(rhobar)
                          
                #-----------------entropies------------
  
                        
                        srhobarc = 0
                        srhobarb = 0
                        srhobar = 0
                        
                        srhobar += entropy(rhobar)
                        srhobarb += entropy(rhobarb)
                        srhobarc +=  pmeas(povm1, w2)*entropy( rhobarc1) + pmeas(povm2, w2)*entropy( rhobarc2)
                        deltabar2 = srhobarb + srhobarc - srhobar
                        if deltabar2<r2:
                            r2=deltabar2
                deltabarr.append(r2)
                ccc  = ((( 1 + -1* max(abs(c1),abs(c2),abs(c3)) )/2) * (math.log2(1 + -1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ) +
                        + (( 1 + 1* max(abs(c1),abs(c2),abs(c3)) )/2) * (math.log2(1 + 1*max(abs(c1),abs(c2),abs(c3)) ) ))


                sm = (0.25*(1-c1-c2-c3)*math.log2(0.25*(1-c1-c2-c3) +10**-15 ) + 0.25*(1-c1+c2+c3)*math.log2(0.25*(1-c1+c2+c3)+10**-15) 
                      + 0.25*(1+c1-c2+c3)*math.log2(0.25*(1+c1-c2+c3)+10**-15) + 0.25*(1+c1+c2-c3)*math.log2(0.25*(1+c1+c2-c3)+10**-15))

                qqq = 2 + sm - ccc
                
                denc = p0*qqq - r2
                discordss1.append(denc)
                c1ss.append(c1)
                c2ss.append(c2)
                c3ss.append(c3)


discords = np.zeros(len(discordss1))           
c1s = np.zeros(len(discordss1))           
c2s = np.zeros(len(discordss1))  
c3s = np.zeros(len(discordss1))           
         
for i in range(len(discords)):
    c1s[i] = c1ss[i]  
    c2s[i] = c2ss[i]  
    c3s[i] = c3ss[i]
    discords[i] = discordss1[i] 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
img = ax.scatter(c1s , c2s , c3s , c=discords , cmap=plt.hot() , s=50 , vmin = 0 , vmax=0.4 )
fig.colorbar(img)

ax.set(xlabel='C1', ylabel='C2', zlabel='c3')

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:",execution_time)
