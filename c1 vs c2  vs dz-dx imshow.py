import numpy as np
from pylab import plot, xlim, ylim, xlabel, ylabel, uniform,randint , imshow
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

#-----------------------------entropy function--------------------

def entropy( s):
    ent = 0
    ev , es = np.linalg.eig(s)
    for i in ev:
        if i>10**(-16):
            ent += -i*math.log2(i)
    return ent

#-------------------------partial trace function----------------

def ptrace(y):
    x = np.zeros([2,2])
    x[0][0] = y[0][0] + y[1][1] 
    x[0][1] = y[0][2] + y[1][3]
    x[1][0] = y[2][0] + y[3][1]
    x[1][1] = y[2][2] + y[3][3]
    return x

#--------------------------povm function--------------------

def povm( pvm1 , state):
    mes =  np.matmul(np.matmul(pvm1, state),np.transpose(np.conjugate(pvm1)))
    
    rhoc = ptrace(mes)/np.trace(np.matmul(np.matmul(np.transpose(np.conjugate(pvm1)),pvm1), state))
    return rhoc


def pmeas(x,y):
    pi = np.trace(np.dot(np.dot(np.transpose(np.conjugate(x)),x), y))
    return pi

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
    x4 = np.dot(np.dot(gama1a , state) , np.transpose(gama1a))
    w =  (x1 + x4)
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
    x4 = np.dot(np.dot(gama1a , state) , np.transpose(gama1a))
    w =  (x1 + x4)
    return w

#--------------------------------------------------------------


states=[]
ff = 0.005   #decoherence rate
eta = 0.5  #reflectivity
p0 = 0.5 #probability of presence of the object
p1 = 1 - p0
r1=5#for minimizing discord
r2=5
dmax=0

#-----------------------------------bitflip-----------
pa = 0.2
l=27 #range for generating states
q = 0 #counter
c3 = 0.6
T1 = 50
T2 = T1  
discords1 = np.zeros([2*l+1 , 2*l+1])  
discords2 = np.zeros([2*l+1 , 2*l+1])  


for i11 in range(-l,l+1):
    for i2 in range(-l,l+1):
        for i3 in range(1):
            

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
                
                
                #states.append( 1/4 * bb)
                    
#-----------------------computing discord of encoding-------------
              
                bell =0.25*bb

                   
                w = bitflip(bell, pa)
                   
                #----------------------------
                rhob = ptrace(w)
                
                
                ###################################### signal returns
                #######################################
                   
                
                w1 = (1-eta)*ii/4 + eta*w
                
                w2 =  bitflip(w1, pa)
                   
                #---------------sub state----------- 
                    
                rhob2 = ptrace(w2)
                 
                
                theta = 0
                phi = 0
                r1,r2=5,5
                for t1 in range(62):
                    for t2 in range(62):
                        theta = t1/10
                        phi = t2/10
                #------------------------------------------------------------------------------------       
                        zero0 = (np.cos(theta))*ket0 + (np.exp(1j*phi))*(np.sin(theta))*ket1
                        zero1 = np.transpose(np.conjugate(zero0))
                        
                        povm1 = np.kron( np.identity(2) , np.outer(zero0, zero1))
                
                        one0 = np.exp((-1j)*phi)*np.sin(theta)*ket0 - np.cos(theta)*ket1 
                        one1 = np.transpose(np.conjugate(one0))
                        
                        povm2 = np.kron( np.identity(2) , np.outer(one0 , one1))
                        
                #----------------------------------------------conditional state------------
                        rhoc21 = povm(povm1, w2)
                        rhoc22 = povm(povm2, w2)
                        rhoc2 = pmeas(povm1, w2)*rhoc21 + pmeas(povm1, w2)*rhoc22
                #-----------------------------------------------average state----------   
                     
                        rhobar = p0 *w2 + p1*ii/4                     
                #------------------------------------------------measurement on average state----                
                        rhobarc1 = p0*rhoc21 + p1*i1/2
                        rhobarc2 = p0*rhoc22 + p1*i1/2                
                #---------------------------------------------------average substate-----------                
                        rhobarb = ptrace(rhobar)                          
                #-----------------entropies------------                                        
                        srhow2 = 0
                        srhob2 = 0
                        srhoc2 = 0 
                        srhow2 += entropy(w2)
                        srhob2 += entropy(rhob2)
                        srhoc2 += pmeas(povm1, w2)*entropy( rhoc21) + pmeas(povm2, w2)*entropy( rhoc22)
                        
                        dd22 = srhoc2 + srhob2 - srhow2 
                        if dd22<r1:
                            r1=dd22
                                                
                        srhobarc = 0
                        srhobarb = 0
                        srhobar = 0
                        
                        srhobar += entropy(rhobar)
                        srhobarb += entropy(rhobarb)
                        srhobarc +=  pmeas(povm1, w2)*entropy( rhobarc1) + pmeas(povm2, w2)*entropy( rhobarc2)
                        deltabar2 = srhobarb + srhobarc - srhobar
                        if deltabar2<r2:
                            r2=deltabar2
                denc2 = p0*r1 - r2
                a1 = int(c1*l+l)
                a2 = int(c2*l+l)
                discords1[a1][a2] = denc2
               

r1=5#for minimizing discord
r2=5

#----------------------phaseflip---------------
for i11 in range(-l,l+1):
    for i2 in range(-l,l+1):
        for i3 in range(1):
            

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
                
                
                
                
                
               # states.append( 1/4 * bb)
                
#-----------------------computing discord of encoding-------------
              
                bell =0.25*bb
                 
                   
                w = phaseflip(bell, pa)
                   
                #----------------------------
                rhob = ptrace(w)
                
                
                ###################################### signal returns
                #######################################
                                   
                w1 = (1-eta)*ii/4 + eta*w
                
                w2 =  phaseflip(w1, pa)
                   
                #---------------sub state----------- 
                    
                rhob2 = ptrace(w2)
                 
                
                theta = 0
                phi = 0
                dd11=0
                dd22=1
                r1,r2=5,5
                for t1 in range(62):
                    for t2 in range(62):
                        theta = t1/10#t1/10 #be ezaye yek theta va phi moshakhas rooye 0 , 1 miangin migirim
                        phi = t2/10#t2/100 # mirim bara theta va phi badi.
                #------------------------------------------------------------------------------------  check         
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
                
                        
                        srhow2 = 0
                        srhob2 = 0
                        srhoc2 = 0 
                        srhow2 += entropy(w2)
                        srhob2 += entropy(rhob2)
                        srhoc2 += pmeas(povm1, w2)*entropy( rhoc21) + pmeas(povm2, w2)*entropy( rhoc22)
                        
                        dd22 = srhoc2 + srhob2 - srhow2 #J max yani srhoc min discord min
                        if dd22<r1:
                            r1=dd22
                        
                        
                        srhobarc = 0
                        srhobarb = 0
                        srhobar = 0
                        
                        srhobar += entropy(rhobar)
                        srhobarb += entropy(rhobarb)
                        srhobarc +=  pmeas(povm1, w2)*entropy( rhobarc1) + pmeas(povm2, w2)*entropy( rhobarc2)
                        deltabar2 = srhobarb + srhobarc - srhobar
                        if deltabar2<r2:
                            r2=deltabar2
                denc2 = p0*r1 - r2
                a1 = int(c1*l+l)
                a2 = int(c2*l+l)
                discords2[a1][a2] = denc2
                
discords = np.zeros([2*l+1 , 2*l+1])
for i in range(2*l+1):
    for j in range(2*l+1):
        discords[i][j] = discords2[i][j] - discords1[i][j]
                
xx=imshow(discords ,interpolation='nearest',origin='lower', vmin=-0.01,vmax=0.01, extent=[-1,1,-1,1])

cb=plt.colorbar(xx) 

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:",execution_time)
