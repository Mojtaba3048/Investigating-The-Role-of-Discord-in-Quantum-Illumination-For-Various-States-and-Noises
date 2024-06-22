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


l=15 #range for generating states
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
                
                ccc  = ((( 1 + -1* max(abs(c1),abs(c2),abs(c3)) )/2) * (math.log2(1 + -1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ) +
                        + (( 1 + 1* max(abs(c1),abs(c2),abs(c3)) )/2) * (math.log2(1 + 1*max(abs(c1),abs(c2),abs(c3)) ) ))


                sm = (0.25*(1-c1-c2-c3)*math.log2(0.25*(1-c1-c2-c3) +10**-15 ) + 0.25*(1-c1+c2+c3)*math.log2(0.25*(1-c1+c2+c3)+10**-15) 
                      + 0.25*(1+c1-c2+c3)*math.log2(0.25*(1+c1-c2+c3)+10**-15) + 0.25*(1+c1+c2-c3)*math.log2(0.25*(1+c1+c2-c3)+10**-15))

                qqq = 2 + sm - ccc
                
                discordss1.append(qqq)
                c1ss.append(c1)
                c2ss.append(c2)
                c3ss.append(c3)



discords = np.zeros(len(discordss1))           
c1s = np.zeros(len(discordss1))           
c2s = np.zeros(len(discordss1))  
c3s = np.zeros(len(discordss1))           
         
for i in range(len(discords)):
    c1s[i] = c1ss[i] #- c1ss2[i] 
    c2s[i] = c2ss[i] #- c2ss2[i] 
    c3s[i] = c3ss[i]
    discords[i] = discordss1[i] #- discordss1[i]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
img = ax.scatter(c1s , c2s , c3s , c=discords , cmap=plt.hot() , s=50 , vmin = 0 , vmax= 1 )
fig.colorbar(img)

ax.set(xlabel='C1', ylabel='C2', zlabel='c3')

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:",execution_time)
