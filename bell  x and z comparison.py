import numpy as np
from pylab import plot, xlim, ylim, xlabel, ylabel
import math
import time

start_time = time.time()

#-----------------------basis--------
bra0 = np.zeros([1,2],dtype = complex)
bra0[0][0]=1

ket0 = np.zeros([2,1],dtype = complex)
ket0[0][0]= 1

bra1 = np.zeros([1,2],dtype = complex)
bra1[0][1]=1

ket1 = np.zeros([2,1],dtype = complex)
ket1[1][0]=1

ket00 = np.kron(ket0, ket0)
bra00 = np.kron(bra0, bra0)
ket11 = np.kron(ket1, ket1)
bra11 = np.kron(bra1, bra1)

#--------------identity--------

ii = np.identity(4)
i1 = np.identity(2)

#-----------------bell state--------

bell1 = (1/(np.sqrt(2))) * (ket00 + ket11)
bell2 = (1/(np.sqrt(2))) * (bra00 + bra11)
bell = np.kron(bell1, bell2) 



#-----------------werner state------
zz=[]
du = 10
ff = 0.005   #decoherence rate
eta = 0.5  #reflectivity
p0 = 0.24
p1 = 1 - p0
dist=[]
df=[]
r1 , r2 , r3 , r4 , r5 , r6 = 5,5,5,5,5,5
#-------------------entropy function--------------------
def entropy( s):
    ent = 0
    ev , es = np.linalg.eig(s)
    for i in ev:
        if i>10**(-16):
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

#--------------------------------------------------------

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

#----------------------------------------------------------
r=5

for ttt in range(du):
    T1 = ttt
    T2 = T1   
    pa = 1 - np.exp(-ff *(T1))
     
#-------------kraus operators on bell state-------- signal send
       
    w = bitflip(bell, pa)
   
#----------------------------
    rhob = ptrace(w)
    

###################################### signal returns
#######################################
   
    pa = 1 - np.exp(-ff *(T2))
    
    w1 = (1-eta)*ii/4 + eta*w
    
    w2 =  bitflip(w1, pa)
       
#---------------sub state----------- 
        
    rhob2 = ptrace(w2)
    
#---------------measurement operator------
    
    theta = 0
    phi = 0
    dd11=0
    dd22=1
    for t1 in range(31):
        for t2 in range(31):
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
            srhoc2 += pmeas(povm1, w)*entropy( rhoc21) + pmeas(povm2, w)*entropy( rhoc22)
            
            dd22 = srhoc2 + srhob2 - srhow2 #J max yani srhoc min discord min
            if dd22<r1:
                r1=dd22
            
            
            srhobarc = 0
            srhobarb = 0
            srhobar = 0
            
            srhobar += entropy(rhobar)
            srhobarb += entropy(rhobarb)
            srhobarc +=  pmeas(povm1, w)*entropy( rhobarc1) + pmeas(povm2, w)*entropy( rhobarc2)
            
            deltabar2 = srhobarb + srhobarc - srhobar
            if deltabar2<r2:
                r2=deltabar2
            
            
            denc2 = p0*r1 - r2
            
            
    dist.append(1 - np.exp(-ff *(ttt)))
    df.append(denc2)


                         
plot(dist , df ,'o',  markersize= '1.5', color = 'b' )

#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================

zz=[]

dist=[]
df=[]

r=5
for ttt in range(du):
    T1 = ttt
    T2 = T1   
    pa = 1 - np.exp(-ff *(T1))
     
#-------------kraus operators on bell state-------- signal send
       
    w = phaseflip(bell, pa)
   
#----------------------------

    rhob = ptrace(w)

###################################### signal returns
#######################################
   
    pa = 1 - np.exp(-ff *(T2))
     
#-------------kraus operators on bell state-------- 
    
    w1 = (1-eta)*ii/4 + eta*w
    
    w2 =  phaseflip(w1, pa)
       
#---------------sub state----------- 
        
    rhob2 = ptrace(w2)
    

    theta = 0
    phi = 0
    dd11=0
    dd22=1
    for t1 in range(1):
        for t2 in range(1):
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
            srhoc2 += pmeas(povm1, w)*entropy( rhoc21) + pmeas(povm2, w)*entropy( rhoc22)
            
            dd22 = srhoc2 + srhob2 - srhow2 #J max yani srhoc min discord min
            if dd22<r3:
                r3=dd22
            
            
            
            srhobarc = 0
            srhobarb = 0
            srhobar = 0
            
            srhobar += entropy(rhobar)
            srhobarb += entropy(rhobarb)
            srhobarc +=  pmeas(povm1, w)*entropy( rhobarc1) + pmeas(povm2, w)*entropy( rhobarc2)           
            deltabar2 = srhobarb + srhobarc - srhobar
            if deltabar2<r4:
                r4=deltabar2
            
            denc2 = p0*r3 - r4
            
    dist.append(1 - np.exp(-ff *(ttt)))
    df.append(denc2)

                         
plot(dist , df , color = 'g')


##########################################
#############################################

#xlim(0, 0.07)
#ylim(0.03, 0.05)
xlabel('green=phase-flip , blue=bit-flip    P(t)')
ylabel('discord of encoding')
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:",execution_time)
